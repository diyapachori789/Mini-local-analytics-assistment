<#
.SYNOPSIS
    Opens the DuckDB Local UI with this project's databases attached read-only.

.DESCRIPTION
    Starts DuckDB's built-in web UI and attaches:

        analytics.duckdb  AS analytics   (opportunities)
        history.duckdb    AS history     (query_history)

    Both are attached READ_ONLY, so nothing here can modify either database.
    The UI's own state needs a writable catalog, so the main connection is an
    in-memory database and the project files are attached to it. A read-only
    main connection makes the UI fail with "Catalog _duckdb_ui does not exist".

    DuckDB reports the port it actually bound; this script uses that URL rather
    than assuming 4213.

    Nothing in the Flask application or the project backend is imported,
    started, or changed.

.PARAMETER NoBrowser
    Print the URL but do not open a browser.

.PARAMETER TimeoutSeconds
    How long to wait for the UI to report its URL. Default 90.

.EXAMPLE
    .\open-duckdb-ui.ps1

.EXAMPLE
    .\open-duckdb-ui.ps1 -NoBrowser

.NOTES
    Press Ctrl+C to stop the viewer. DuckDB locks a database file while it is
    attached, so close this viewer before using the Flask app's history, and
    stop the Flask app before opening analytics.duckdb here.
#>

[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [ValidateRange(5, 600)]
    [int]$TimeoutSeconds = 90
)

$ErrorActionPreference = 'Stop'

# $PSScriptRoot keeps every path relative to this file, so a project directory
# containing spaces is handled without any quoting by the caller.
$projectRoot = $PSScriptRoot
$analyticsDb = Join-Path -Path $projectRoot -ChildPath 'analytics.duckdb'
$historyDb   = Join-Path -Path $projectRoot -ChildPath 'history.duckdb'

Write-Host ''
Write-Host 'DuckDB Local UI' -ForegroundColor Cyan
Write-Host ('  project : {0}' -f $projectRoot)

# --- Locate a Python interpreter ------------------------------------------

$venvPython = Join-Path -Path $projectRoot -ChildPath '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $venvPython) {
    $python = $venvPython
} else {
    $fallback = Get-Command python -ErrorAction SilentlyContinue
    if (-not $fallback) {
        Write-Host ''
        Write-Host 'ERROR: no Python interpreter found.' -ForegroundColor Red
        Write-Host '  Expected the project virtual environment at:'
        Write-Host ("    {0}" -f $venvPython)
        Write-Host '  Create it, or make "python" available on PATH.'
        exit 1
    }
    $python = $fallback.Source
    Write-Host '  note    : project .venv not found, using python from PATH' -ForegroundColor Yellow
}
Write-Host ('  python  : {0}' -f $python)

# --- Check the database files exist ---------------------------------------

$missing = @()
foreach ($database in @($analyticsDb, $historyDb)) {
    if (-not (Test-Path -LiteralPath $database)) { $missing += $database }
}
if ($missing.Count -gt 0) {
    Write-Host ''
    Write-Host 'ERROR: database file(s) not found:' -ForegroundColor Red
    $missing | ForEach-Object { Write-Host ("    {0}" -f $_) }
    Write-Host ''
    Write-Host '  analytics.duckdb is created by running the app once:'
    Write-Host '    .\.venv\Scripts\python.exe app.py'
    Write-Host '  history.duckdb is created the first time a question is saved.'
    exit 1
}

# --- Warn early when the Flask app is holding a database --------------------

$flask = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='py.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match 'web_app\.py' }
if ($flask) {
    Write-Host ''
    Write-Host 'NOTE: the Flask app is running and holds analytics.duckdb.' -ForegroundColor Yellow
    $flask | ForEach-Object { Write-Host ('    PID {0}' -f $_.ProcessId) }
    Write-Host '  DuckDB cannot attach a database another process has open,'
    Write-Host '  so the attach below will most likely fail. Stop the Flask app first.'
}

# --- Write the helper that actually runs DuckDB ----------------------------

$helper = Join-Path -Path $env:TEMP -ChildPath ('duckdb_ui_{0}.py' -f $PID)

$helperSource = @'
"""Attach the project databases read-only and serve the DuckDB Local UI."""
import sys
import time

import duckdb

analytics_path, history_path = sys.argv[1], sys.argv[2]

# The UI stores its own state, so the main connection must be writable.
# An in-memory database satisfies that without touching any project file.
connection = duckdb.connect()

try:
    connection.execute("INSTALL ui")
    connection.execute("LOAD ui")
except Exception as exc:
    print("ERROR|ui_extension|%s" % str(exc).splitlines()[0], flush=True)
    sys.exit(3)

for alias, path in (("analytics", analytics_path), ("history", history_path)):
    # ATTACH takes a literal path, not a bound parameter, so the value is
    # escaped by doubling any single quote before it is embedded.
    literal = path.replace("'", "''")
    try:
        connection.execute("ATTACH '%s' AS %s (READ_ONLY)" % (literal, alias))
    except Exception as exc:
        detail = " ".join(str(exc).split())
        locked = "another process" in detail or "already open" in detail
        print("ERROR|attach_%s|%s|%s" % (alias, "locked" if locked else "other", detail), flush=True)
        sys.exit(4)

# Refuse to serve unless the attachments really are read-only.
for alias in ("analytics", "history"):
    try:
        connection.execute("CREATE TABLE %s.__write_probe (x INTEGER)" % alias)
        print("ERROR|not_readonly|%s is writable" % alias, flush=True)
        sys.exit(5)
    except duckdb.Error:
        pass

databases = sorted(row[0] for row in connection.execute("SHOW DATABASES").fetchall())
print("DATABASES|%s" % ",".join(databases), flush=True)
for alias, table in (("analytics", "opportunities"), ("history", "query_history")):
    try:
        rows = connection.execute("SELECT COUNT(*) FROM %s.%s" % (alias, table)).fetchone()[0]
        print("TABLE|%s.%s|%s" % (alias, table, rows), flush=True)
    except Exception:
        print("TABLE|%s.%s|unavailable" % (alias, table), flush=True)

try:
    started = connection.execute("CALL start_ui_server()").fetchall()
except Exception as exc:
    print("ERROR|ui_server|%s" % str(exc).splitlines()[0], flush=True)
    sys.exit(6)

# Use the address DuckDB reports rather than assuming a port.
import re

message = str(started[0][0]) if started and started[0] else ""
match = re.search(r"https?://[^\s'\"]+", message)
url = match.group(0).rstrip("/") if match else ""
if not url:
    print("ERROR|ui_url|could not determine the UI address", flush=True)
    sys.exit(7)

print("URL|%s" % url, flush=True)
print("READY", flush=True)

while True:
    time.sleep(3600)
'@

Set-Content -LiteralPath $helper -Value $helperSource -Encoding UTF8

# --- Start the viewer ------------------------------------------------------

$stdout = Join-Path -Path $env:TEMP -ChildPath ('duckdb_ui_{0}.out' -f $PID)
$stderr = Join-Path -Path $env:TEMP -ChildPath ('duckdb_ui_{0}.err' -f $PID)
foreach ($file in @($stdout, $stderr)) {
    if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file -Force }
}

Write-Host ''
Write-Host 'Starting DuckDB UI (attaching both databases read-only)...'

# Start-Process joins ArgumentList with spaces and does not quote, so any path
# containing a space (this project directory does) must be quoted explicitly or
# it arrives at Python split across several argv entries.
$arguments = @(
    '-u'
    ('"{0}"' -f $helper)
    ('"{0}"' -f $analyticsDb)
    ('"{0}"' -f $historyDb)
)

$process = Start-Process -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $projectRoot `
    -NoNewWindow -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr

$url = $null
$attached = $null
$tableSummary = [ordered]@{}
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)

try {
    while ((Get-Date) -lt $deadline) {
        if (Test-Path -LiteralPath $stdout) {
            $lines = Get-Content -LiteralPath $stdout -ErrorAction SilentlyContinue

            $failure = $lines | Where-Object { $_ -like 'ERROR|*' } | Select-Object -First 1
            if ($failure) {
                $parts = $failure -split '\|', 4
                $stage = $parts[1]
                # Attach failures carry a kind marker; other stages do not.
                $kind   = if ($parts.Count -ge 4) { $parts[2] } else { 'other' }
                $detail = if ($parts.Count -ge 4) { $parts[3] } else { $parts[2] }

                Write-Host ''
                Write-Host 'Could not open the DuckDB UI.' -ForegroundColor Red
                Write-Host ('  stage  : {0}' -f $stage)
                Write-Host ('  detail : {0}' -f $detail)

                if ($kind -eq 'locked') {
                    Write-Host ''
                    Write-Host '  That database is open in another process. DuckDB allows only' -ForegroundColor Yellow
                    Write-Host '  one process per file, even read-only. Close the holder and retry:'
                    Write-Host '    - the Flask app        (stop the window running web_app.py)'
                    Write-Host '    - another DuckDB UI    (close its terminal)'
                    Write-Host '    - a Python or pytest session with the database open'
                }
                Write-Host ''
                Write-Host '  Nothing was modified.' -ForegroundColor Yellow
                exit 1
            }

            # Collected rather than printed here: the file is re-read on every
            # poll, so printing inside the loop would repeat each line.
            foreach ($line in $lines) {
                if ($line -like 'DATABASES|*') { $attached = ($line -split '\|', 2)[1] }
                if ($line -like 'TABLE|*') {
                    $parts = $line -split '\|', 3
                    $tableSummary[$parts[1]] = $parts[2]
                }
                if ($line -like 'URL|*') { $url = ($line -split '\|', 2)[1] }
            }
            if ($url) { break }
        }

        if ($process.HasExited) {
            Write-Host ''
            Write-Host ('DuckDB UI exited early (code {0}).' -f $process.ExitCode) -ForegroundColor Red
            if (Test-Path -LiteralPath $stderr) {
                Get-Content -LiteralPath $stderr | Select-Object -First 15 | ForEach-Object { Write-Host "  $_" }
            }
            exit 1
        }

        Start-Sleep -Milliseconds 400
    }

    if (-not $url) {
        Write-Host ''
        Write-Host ('Timed out after {0}s waiting for the UI to start.' -f $TimeoutSeconds) -ForegroundColor Red
        Write-Host '  The ui extension may still be downloading on first use. Retry,'
        Write-Host '  or raise the wait:  .\open-duckdb-ui.ps1 -TimeoutSeconds 300'
        exit 1
    }

    if ($attached) {
        Write-Host ('  attached : {0}' -f $attached) -ForegroundColor Green
    }
    foreach ($entry in $tableSummary.GetEnumerator()) {
        Write-Host ('  {0,-26} {1} rows' -f $entry.Key, $entry.Value)
    }

    Write-Host ''
    Write-Host '  Both databases are attached READ-ONLY. Nothing can be modified.' -ForegroundColor Green
    Write-Host ''
    Write-Host '  DuckDB UI is running at:' -ForegroundColor Cyan
    Write-Host ("      {0}" -f $url) -ForegroundColor White
    Write-Host ''
    Write-Host '  Try:  USE analytics;  SELECT * FROM opportunities LIMIT 10;'
    Write-Host '        USE history;    SELECT * FROM query_history ORDER BY created_at DESC;'
    Write-Host ''

    if (-not $NoBrowser) {
        try {
            Start-Process $url | Out-Null
            Write-Host '  Opened in your default browser.'
        } catch {
            Write-Host '  Could not open a browser automatically; use the URL above.' -ForegroundColor Yellow
        }
    }

    Write-Host ''
    Write-Host '  Press Ctrl+C to stop the viewer and release both databases.' -ForegroundColor Yellow
    Wait-Process -Id $process.Id
}
finally {
    # Always release the database locks, including on Ctrl+C.
    if ($process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        Write-Host ''
        Write-Host 'DuckDB UI stopped. Both databases released.' -ForegroundColor Cyan
    }
    foreach ($file in @($helper, $stdout, $stderr)) {
        if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file -Force -ErrorAction SilentlyContinue }
    }
}
