"""The browser-visible shell: title, top controls, composer, attachment.

These assert the served markup and script, which is what a person actually
gets. The backend contract is checked too, because the attachment control only
makes sense alongside what the API will and will not accept.
"""

from __future__ import annotations

import re

import pytest

import web_app


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(web_app, "CHARTS_DIR", tmp_path, raising=False)
    application = web_app.create_app(process_question_func=lambda *a, **k: None)
    application.config.update(TESTING=True, CHARTS_DIR=tmp_path)
    return application.test_client()


@pytest.fixture
def page(client):
    return client.get("/").get_data(as_text=True)


@pytest.fixture
def script(client):
    return client.get("/static/js/app.js").get_data(as_text=True)


@pytest.fixture
def styles(client):
    return client.get("/static/css/app.css").get_data(as_text=True)


class TestBranding:
    def test_the_visible_title_is_exact(self, page):
        assert '<h1 class="app-title">ANALYTICS ASSISTANT</h1>' in page

    def test_only_one_h1(self, page):
        assert len(re.findall(r"<h1\b", page)) == 1

    @pytest.mark.parametrize(
        "banned",
        ["Mini Local Analytics Assistant", "Analytics Workspace",
         "Ask, explore, and follow up"],
    )
    def test_rejected_names_are_absent(self, page, banned):
        assert banned not in page

    def test_the_marketing_subtitle_is_gone(self, page):
        assert "Grounded answers and useful local charts in one saved conversation." not in page
        assert 'class="topbar-subtitle"' not in page


class TestMenuRail:
    """Every secondary control lives in the rail, not above the conversation."""

    @pytest.mark.parametrize(
        "marker",
        ["data-new-chat", 'data-view-target="about"', 'id="theme-toggle"',
         "data-chart-shortcut", 'data-view-target="chat"'],
    )
    def test_each_control_sits_in_the_rail(self, page, marker):
        rail = page.split('class="menu-rail"')[1].split("</nav>")[0]
        assert marker in rail

    def test_the_conversation_area_has_no_utility_row(self, page):
        assert 'class="topbar-actions"' not in page
        main = page.split('<main class="main-content"')[1]
        for control in ("data-new-chat", 'data-view-target="about"', 'id="theme-toggle"'):
            assert control not in main

    def test_the_rail_controls_are_named_for_screen_readers(self, page):
        rail = page.split('class="menu-rail"')[1].split("</nav>")[0]
        for label in ("New chat", "Chats", "Charts", "About"):
            assert f"<span class=\"sr-only\">{label}</span>" in rail


class TestCountersRemoved:
    def test_no_saved_counter_anywhere(self, page, script):
        assert "data-saved-chats" not in page
        assert 'id="conversation-count"' not in page
        assert "saved-button-label" not in page
        assert "conversationCount" not in script

    def test_no_visible_messages_saved_locally_text(self, page):
        assert "messages saved locally" not in page
        assert "message saved locally" not in page

    def test_the_conversation_meta_line_is_not_visible(self, page):
        """Kept in the DOM for announcements, hidden from the reading order."""
        header = page.split('class="conversation-header"')[1].split("</header>")[0]
        assert 'class="conversation-meta sr-only"' in header

    def test_loading_a_chat_shows_no_success_toast(self, script):
        assert "Saved chat loaded." not in script
        assert 'setQuestionMessage("Saved chat loaded.", "success")' not in script


class TestNoDuplicateActions:
    def test_new_chat_appears_once(self, page):
        assert page.count("data-new-chat") == 1

    def test_about_appears_once(self, page):
        assert page.count('data-view-target="about"') == 1

    def test_the_history_panel_holds_conversations_only(self, page):
        """The list beside the rail is for browsing chats and nothing else."""
        panel = page.split('class="conversation-sidebar-section"')[1].split("</section>")[0]
        for control in ("data-new-chat", 'data-view-target="about"',
                        'id="theme-toggle"', "data-chart-shortcut",
                        'id="conversation-count"'):
            assert control not in panel
        assert 'id="conversation-list"' in panel


class TestDeleteChatUx:
    def test_delete_is_not_rendered_permanently_per_row(self, script):
        assert 'deleteButton.textContent = "Delete";' not in script
        assert "conversation-menu-trigger" in script

    def test_each_row_gets_an_overflow_menu(self, script):
        assert 'trigger.setAttribute("aria-haspopup", "true")' in script
        assert 'trigger.setAttribute("aria-expanded", "false")' in script
        assert "Chat actions" in script

    def test_the_menu_closes_on_escape_and_outside_click(self, script):
        assert "closeConversationMenus" in script
        assert 'document.addEventListener("click", closeConversationMenus)' in script

    def test_deletion_still_confirms(self, script):
        assert "function deleteConversation" in script
        assert "confirm(" in script

    def test_the_trigger_is_reachable_without_a_mouse(self, styles):
        assert ".conversation-list-item:focus-within .conversation-menu-trigger" in styles

    def test_the_about_section_itself_is_kept(self, page):
        assert 'id="view-about"' in page
        assert "Local analytics, clear boundaries." in page


class TestComposer:
    def test_the_textarea_and_send_button_are_unchanged(self, page):
        assert 'id="question-input"' in page
        assert 'id="ask-button"' in page
        assert 'maxlength="2000"' in page

    def test_enter_to_send_is_still_documented(self, page, script):
        assert "Press Enter to send. Use Shift and Enter for a new line." in page
        assert "submitOnEnter" in script

    def test_no_manual_chart_controls_returned(self, page):
        for banned in ("chart-toggle", "chart-type", "preferred chart", "Visualization"):
            assert banned not in page


class TestImageUpload:
    def test_the_file_input_exists_and_is_labelled(self, page):
        assert 'id="image-input"' in page
        assert 'type="file"' in page
        assert 'for="image-input"' in page
        assert "Attach an image" in page

    def test_only_image_types_are_accepted(self, page):
        accept = re.search(r'id="image-input"[^>]*accept="([^"]+)"', page)
        assert accept, "the file input must constrain its types"
        assert set(accept.group(1).split(",")) == {"image/png", "image/jpeg", "image/webp"}

    def test_the_type_check_is_repeated_in_script(self, script):
        """An accept attribute is a filter in the picker, not a guarantee."""
        assert "ALLOWED_IMAGE_TYPES" in script
        assert "image/webp" in script

    def test_a_preview_chip_and_remove_control_exist(self, page):
        assert 'id="attachment-chip"' in page
        assert 'id="attachment-thumb"' in page
        assert 'id="attachment-remove"' in page
        assert "Remove attached image" in page

    def test_the_chip_starts_hidden(self, page):
        chip = page.split('id="attachment-chip"')[1].split(">")[0]
        assert "hidden" in chip

    def test_the_filename_is_rendered_as_text(self, script):
        assert "ui.attachmentName.textContent = file.name" in script

    def test_the_attachment_can_be_removed_and_state_reset(self, script):
        assert "function clearAttachment()" in script
        assert 'ui.imageInput.value = ""' in script
        assert "URL.revokeObjectURL" in script, "the object URL must not leak"

    def test_sending_with_an_attachment_is_refused_honestly(self, script):
        assert "Image analysis is not supported yet." in script
        assert "attachedImageName !== null" in script

    def test_the_backend_would_reject_an_image_field(self, client):
        """The refusal above is not conservatism - the API really has no route."""
        response = client.post(
            "/api/query", json={"question": "hi", "image": "data:image/png;base64,xx"}
        )
        assert response.status_code == 400

    def test_the_api_accepts_no_file_field(self):
        source = open("web_app.py", encoding="utf-8").read()
        assert '{"question", "chart_requested", "chart_type", "conversation_id"}' in source
        assert "request.files" not in source


class TestFrontendSafety:
    @pytest.mark.parametrize(
        "banned", ["innerHTML", "outerHTML", "eval(", "new Function", "document.write"]
    )
    def test_no_unsafe_dom_api(self, script, banned):
        assert banned not in script

    def test_markup_has_no_duplicate_ids(self, page):
        ids = re.findall(r'id="([^"]+)"', page)
        assert len(ids) == len(set(ids)), sorted(i for i in ids if ids.count(i) > 1)

    def test_every_script_lookup_has_a_target(self, page, script):
        ids = set(re.findall(r'id="([^"]+)"', page))
        wanted = set(re.findall(r'getElementById\("([^"]+)"\)', script))
        assert wanted <= ids, sorted(wanted - ids)


class TestAccessibility:
    def test_the_file_input_is_reachable_and_named(self, page):
        assert 'class="sr-only" type="file"' in page, "hidden from view, not from the tab order"
        assert 'for="image-input"' in page

    def test_controls_are_semantic_buttons(self, page):
        rail = page.split('class="menu-rail"')[1].split("</nav>")[0]
        assert rail.count("<button") >= 5
        assert "<a " not in rail

    def test_state_changes_are_announced(self, script):
        assert 'announce("Image attached: "' in script

    def test_the_remove_control_has_screen_reader_text(self, page):
        assert "Remove attached image" in page

    def test_focus_moves_sensibly_after_removal(self, script):
        assert "ui.questionInput.focus()" in script


class TestThemingAndResponsiveness:
    def test_new_styles_use_theme_tokens_not_fixed_colours(self, styles):
        block = styles.split("/* --- Composer attachment")[1]
        assert "var(--" in block
        # #fff on a saturated accent is theme-independent, and is what the rest
        # of this sheet already does. A hard-coded surface or text colour is the
        # real problem, because it survives the theme switch unchanged.
        hard_coded = [
            match.group(0)
            for match in re.finditer(r":\s*#[0-9a-fA-F]{3,6}\b", block)
            if match.group(0).split("#")[1].lower() not in ("fff", "ffffff")
        ]
        assert not hard_coded, hard_coded

    def test_top_controls_collapse_on_small_screens(self, styles):
        assert "@media (max-width: 720px)" in styles
        assert "new-chat-label" in styles
        assert "saved-button-label" in styles

    def test_the_title_does_not_wrap(self, styles):
        assert ".app-title" in styles
        assert "white-space: nowrap" in styles.split(".app-title")[1].split("}")[0]


class TestDesktopProportions:
    """The conversation must own the screen, and share one column with the
    header and composer. Measured for real at 1920/1440/1366 as well."""

    def test_one_shared_content_column(self, styles):
        block = styles.split("Desktop shell, rebuilt around ONE content column")[1]
        assert "--content-width" in block
        for selector in (".topbar", ".conversation-stream", ".composer-card"):
            assert selector in block

    def test_the_workspace_is_not_capped(self, styles):
        block = styles.split("Desktop shell, rebuilt around ONE content column")[1]
        assert "max-width: none" in block

    def test_the_title_outranks_the_hero_rule(self, styles):
        """`.topbar h1` beat `.app-title` on specificity and kept it at 2.2rem."""
        assert ".topbar h1.app-title" in styles

    def test_assistant_messages_use_the_whole_column(self, styles):
        block = styles.split(".chat-message-assistant {")[-1].split("}")[0]
        assert "width: 100%" in block

    def test_charts_have_room(self, styles):
        block = styles.split(".message-chart {")[-1].split("}")[0]
        assert "820px" in block

    def test_tables_scroll_inside_themselves(self, styles):
        block = styles.split(".message-data-panel {")[-1].split("}")[0]
        assert "overflow-x: auto" in block


class TestDestructiveActionPlacement:
    def test_delete_all_is_not_beside_the_chat_list(self, page):
        panel = page.split('class="conversation-sidebar-section"')[1].split("</section>")[0]
        assert "data-delete-all-conversations" not in panel

    def test_delete_all_lives_in_about(self, page):
        about = page.split('id="view-about"')[1]
        assert "data-delete-all-conversations" in about
        assert "cannot be undone" in about

    def test_no_stuck_loading_status_under_the_composer(self, script):
        assert 'setQuestionMessage("Loading saved chat...", "")' not in script
