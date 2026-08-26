"""
``janus init`` — offline tests.

The wizard writes two files a user will not read closely: a policy that decides
what their agent may do, and an edit to a settings file that may already hold
someone else's configuration. So the tests below are mostly about two
properties — the policy it emits actually denies what it claims to, and the
settings merge never destroys what it did not write.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from janus.cli.claude_settings import (
    SettingsError,
    find_janus_hooks,
    janus_hook_commands,
    load_settings,
    merge_permissions_deny,
    missing_deny_entries,
    remove_permissions_deny,
    settings_diff,
    upsert_janus_hook,
    write_settings,
)
from janus.cli.starter_policy import (
    ALLOW_PRIORITY,
    CLAUDE_CODE_TOOL_DEFS,
    DENY_PRIORITY,
    PLAIN_ALLOW_TOOLS,
    build_starter_policy,
    pattern_for_entry,
)
from janus.policy.loader import parse_policy

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_JSON = REPO_ROOT / "examples" / "claude_code" / "policy.starter.json"
SETTINGS_FIXTURES = Path(__file__).parent / "fixtures" / "claude_settings"

HOOK_COMMAND = "janus-hook pre --policy /etc/janus/policy.json --mode gate"


def settings_fixture(name: str) -> dict:
    return json.loads((SETTINGS_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


class TestStarterPolicy:
    def test_defaults_match_the_shipped_starter_file(self):
        """The docs tell people to copy `policy.starter.json`; the wizard builds
        the same thing in code. Two sources of truth that drift is how a user
        ends up with a policy the docs do not describe."""
        assert parse_policy(build_starter_policy()) == parse_policy(STARTER_JSON)

    def test_every_rule_is_full_form(self):
        """`parse_policy` reads a bare dict as *shorthand conditions* unless it
        carries a rule key — so a tool with an argument named `priority` would
        silently parse as an unconditional allow. Full form is immune."""
        for tool, rules in build_starter_policy().items():
            for rule in rules:
                assert set(rule) == {"priority", "effect", "conditions", "fallback"}, (
                    f"{tool} rule is not full form: {rule}"
                )

    def test_guarded_tools_deny_first_then_allow(self):
        policy = build_starter_policy()
        for tool in ("Read", "Bash", "Write", "Edit", "MultiEdit"):
            rules = policy[tool]
            assert [r["priority"] for r in rules] == [DENY_PRIORITY, ALLOW_PRIORITY]
            assert [r["effect"] for r in rules] == [1, 0]
            assert rules[-1]["conditions"] == {}, "missing trailing allow: tool is dead"

    def test_bypass_enumeration_is_present(self):
        """Gate mode promotes to strict default-deny under bypassPermissions,
        where an unlisted tool is denied rather than deferred."""
        policy = build_starter_policy()
        for tool in PLAIN_ALLOW_TOOLS:
            assert tool in policy, f"{tool} missing: bypass sessions would break"

    def test_extra_secret_patterns_reach_the_read_deny(self):
        policy = build_starter_policy(extra_secret_patterns=["secrets/", "*.key"])
        pattern = policy["Read"][0]["conditions"]["file_path"]["pattern"]
        assert "secrets/" in pattern
        assert r"\.key$" in pattern

    def test_network_and_git_push_toggles_reach_the_bash_deny(self):
        default = build_starter_policy()["Bash"][0]["conditions"]["command"]["pattern"]
        assert "git" not in default

        hardened = build_starter_policy(deny_network_commands=True, deny_git_push=True)
        pattern = hardened["Bash"][0]["conditions"]["command"]["pattern"]
        assert "curl|wget|ssh|scp|nc|telnet" in pattern
        assert r"git\s+push" in pattern

    def test_deny_webfetch_keeps_the_tool_listed(self):
        """A denied tool must stay enumerated: under bypassPermissions an
        unlisted tool reports 'not listed in the policy', which reads like a
        misconfiguration rather than the deny it is."""
        policy = build_starter_policy(deny_webfetch=True)
        assert "WebFetch" in policy
        assert policy["WebFetch"] == [{"priority": 1, "effect": 1, "conditions": {}, "fallback": 0}]

    def test_output_round_trips_through_the_loader(self):
        internal = parse_policy(build_starter_policy(extra_secret_patterns=["a.b(c)"]))
        assert internal["Read"][0][1] == 1
        assert internal["Read"][1][1] == 0


class TestPatternForEntry:
    def test_metacharacters_are_escaped(self):
        """A filename is a literal. If `.` stayed a metacharacter, `prod.env`
        would also match `prodXenv` — and users type filenames, not regexes."""
        assert pattern_for_entry("config/prod.yaml") == r"config/prod\.yaml"
        assert pattern_for_entry("a(b)c") == r"a\(b\)c"

    def test_suffix_glob_becomes_an_anchored_suffix(self):
        assert pattern_for_entry("*.key") == r"\.key$"

    def test_blank_entries_are_dropped(self):
        assert pattern_for_entry("   ") == ""


class TestToolDefs:
    def test_every_policy_tool_has_a_definition(self):
        """The tool table backs the lint step; a tool in the policy but missing
        here would produce a spurious 'unknown tool' warning on a good policy."""
        defined = {t["name"] for t in CLAUDE_CODE_TOOL_DEFS}
        assert set(build_starter_policy()) <= defined

    def test_conditioned_arguments_exist_in_the_table(self):
        from janus.policy.loader import validate_policy_structure

        warnings = validate_policy_structure(
            parse_policy(build_starter_policy()), CLAUDE_CODE_TOOL_DEFS
        )
        assert warnings == [], warnings

    def test_table_shape_is_what_consumers_expect(self):
        for tool in CLAUDE_CODE_TOOL_DEFS:
            assert set(tool) == {"name", "description", "args"}
            assert isinstance(tool["args"], dict)


def test_starter_file_is_valid_json():
    json.loads(STARTER_JSON.read_text(encoding="utf-8"))


class TestSettingsLoad:
    def test_missing_file_is_empty_settings(self, tmp_path):
        assert load_settings(tmp_path / "nope.json") == {}

    def test_empty_file_is_empty_settings(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text("   \n")
        assert load_settings(path) == {}

    def test_malformed_json_raises_rather_than_being_repaired(self, tmp_path):
        """Guessing at a broken config and rewriting it is how a tool eats
        someone's settings. Stop and let them fix it."""
        path = tmp_path / "settings.json"
        path.write_text('{"hooks": {,}')
        with pytest.raises(SettingsError) as exc:
            load_settings(path)
        assert "strict JSON" in str(exc.value)
        assert "nothing has been written" in str(exc.value)

    def test_non_object_json_raises(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text("[]")
        with pytest.raises(SettingsError):
            load_settings(path)


class TestUpsertJanusHook:
    def test_creates_the_hooks_block_from_empty_settings(self):
        result = upsert_janus_hook({}, command=HOOK_COMMAND)
        entry = result["hooks"]["PreToolUse"][0]["hooks"][0]
        assert entry == {"type": "command", "command": HOOK_COMMAND, "timeout": 10}

    def test_timeout_is_always_written(self):
        """The CLI's hook timeout fails OPEN — a deny arriving after it is
        discarded. An entry without an explicit timeout leaves the shim's
        5s deadline racing an unknown budget."""
        result = upsert_janus_hook({}, command=HOOK_COMMAND)
        assert result["hooks"]["PreToolUse"][0]["hooks"][0]["timeout"] == 10

    def test_upserting_twice_is_the_same_as_once(self):
        """Re-running the wizard must update the hook, not stack a second one:
        two enforcement hooks on one seam decide every call twice."""
        once = upsert_janus_hook({}, command=HOOK_COMMAND)
        twice = upsert_janus_hook(once, command=HOOK_COMMAND)
        assert once == twice
        assert len(find_janus_hooks(twice)) == 1

    def test_stale_command_is_replaced_in_place(self):
        before = settings_fixture("stale-janus")
        after = upsert_janus_hook(before, command=HOOK_COMMAND)
        assert janus_hook_commands(after) == [HOOK_COMMAND]
        assert len(after["hooks"]["PreToolUse"]) == 1

    def test_foreign_hooks_are_left_alone(self):
        before = settings_fixture("foreign-hooks")
        after = upsert_janus_hook(before, command=HOOK_COMMAND)

        assert after["hooks"]["PreToolUse"][0] == before["hooks"]["PreToolUse"][0]
        assert after["hooks"]["PostToolUse"] == before["hooks"]["PostToolUse"]
        assert after["permissions"] == before["permissions"]
        assert after["model"] == "opus"
        assert janus_hook_commands(after) == [HOOK_COMMAND]

    def test_duplicate_janus_entries_collapse_to_one(self):
        before = settings_fixture("dup-janus")
        assert len(find_janus_hooks(before)) == 2

        after = upsert_janus_hook(before, command=HOOK_COMMAND)
        assert janus_hook_commands(after) == [HOOK_COMMAND]

        surviving = [
            entry["command"] for group in after["hooks"]["PreToolUse"] for entry in group["hooks"]
        ]
        assert "keep-me pre" in surviving, "a foreign hook was collateral damage"

    def test_input_is_not_mutated(self):
        before = settings_fixture("foreign-hooks")
        snapshot = json.dumps(before, sort_keys=True)
        upsert_janus_hook(before, command=HOOK_COMMAND)
        assert json.dumps(before, sort_keys=True) == snapshot

    def test_hooks_of_the_wrong_type_raise(self):
        with pytest.raises(SettingsError):
            upsert_janus_hook({"hooks": []}, command=HOOK_COMMAND)


class TestPermissionsDeny:
    def test_union_preserves_user_entries_and_order(self):
        before = settings_fixture("foreign-hooks")
        after = merge_permissions_deny(before, ["WebFetch", "Bash(curl:*)"])
        assert after["permissions"]["deny"] == [
            "Bash(rm -rf:*)",
            "WebFetch",
            "Bash(curl:*)",
        ]
        assert after["permissions"]["allow"] == ["Read"]

    def test_merge_is_idempotent(self):
        once = merge_permissions_deny({}, ["WebFetch"])
        twice = merge_permissions_deny(once, ["WebFetch"])
        assert once == twice == {"permissions": {"deny": ["WebFetch"]}}

    def test_missing_entries_reports_only_the_gap(self):
        settings = merge_permissions_deny({}, ["WebFetch"])
        assert missing_deny_entries(settings, ["WebFetch", "Bash(nc:*)"]) == ["Bash(nc:*)"]

    def test_removal_is_explicit_and_narrow(self):
        settings = merge_permissions_deny({}, ["WebFetch", "Bash(git push:*)"])
        after = remove_permissions_deny(settings, ["Bash(git push:*)"])
        assert after["permissions"]["deny"] == ["WebFetch"]


class TestSettingsDiff:
    def test_identical_settings_produce_no_diff(self):
        assert settings_diff({"a": 1}, {"a": 1}, "settings.json") == ""

    def test_diff_names_the_file_and_shows_the_addition(self):
        before: dict = {}
        after = upsert_janus_hook(before, command=HOOK_COMMAND)
        diff = settings_diff(before, after, "settings.json")
        assert "settings.json (current)" in diff
        assert HOOK_COMMAND in diff


def run_init(argv, monkeypatch, capsys, *, stdin="", tty=True):
    """Drive `janus init` in-process, the way the shim tests drive janus-hook."""
    from janus.cli import init as init_module
    from janus.cli.main import main

    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    monkeypatch.setattr(init_module, "stdin_is_tty", lambda: tty)
    code = main(argv)
    return code, capsys.readouterr().out


def project(tmp_path: Path, *, home: Path | None = None, monkeypatch=None) -> Path:
    """A scratch project directory with a relocated home."""
    proj = tmp_path / "proj"
    (proj / ".claude").mkdir(parents=True)
    (proj / "README.md").write_text("hi")
    if monkeypatch is not None:
        from janus.cli import init as init_module

        monkeypatch.setattr(init_module, "_home_dir", lambda: home or (tmp_path / "home"))
    return proj


class TestUmbrellaDispatch:
    def test_help_exits_zero(self, capsys):
        from janus.cli.main import main

        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0

    def test_unknown_subcommand_exits_two(self):
        from janus.cli.main import main

        with pytest.raises(SystemExit) as exc:
            main(["nope"])
        assert exc.value.code == 2

    def test_missing_subcommand_exits_two(self):
        from janus.cli.main import main

        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2

    def test_doctor_delegates_to_the_shim(self, capsys):
        from janus.cli.main import main

        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "payload round-trip: ok" in out
        assert "phase-1 stateless" in out


class TestNonInteractive:
    def test_yes_writes_a_working_deployment(self, tmp_path, monkeypatch, capsys):
        proj = project(tmp_path, monkeypatch=monkeypatch)
        code, out = run_init(["init", "--yes", "--project-dir", str(proj)], monkeypatch, capsys)
        assert code == 0, out

        policy_path = proj / ".claude" / "janus" / "policy.json"
        settings_path = proj / ".claude" / "settings.local.json"
        assert policy_path.exists()

        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        entry = settings["hooks"]["PreToolUse"][0]["hooks"][0]
        assert "janus" in entry["command"]
        assert entry["timeout"] == 10
        assert "WebFetch" in settings["permissions"]["deny"]

    def test_non_tty_without_yes_refuses_and_writes_nothing(self, tmp_path, monkeypatch, capsys):
        """Piped stdin means the 'defaults' were nobody's decision. A security
        tool must not configure itself off an absent human."""
        proj = project(tmp_path, monkeypatch=monkeypatch)
        code, out = run_init(["init", "--project-dir", str(proj)], monkeypatch, capsys, tty=False)
        assert code == 2
        assert "--yes" in out
        assert not (proj / ".claude" / "janus").exists()
        assert not (proj / ".claude" / "settings.local.json").exists()

    def test_dry_run_shows_the_diff_and_writes_nothing(self, tmp_path, monkeypatch, capsys):
        proj = project(tmp_path, monkeypatch=monkeypatch)
        code, out = run_init(
            ["init", "--yes", "--dry-run", "--project-dir", str(proj)], monkeypatch, capsys
        )
        assert code == 0
        assert "Dry run" in out
        assert "PreToolUse" in out
        assert not (proj / ".claude" / "janus").exists()
        assert not (proj / ".claude" / "settings.local.json").exists()

    def test_rerun_is_idempotent(self, tmp_path, monkeypatch, capsys):
        proj = project(tmp_path, monkeypatch=monkeypatch)
        argv = ["init", "--yes", "--project-dir", str(proj), "--force"]
        run_init(argv, monkeypatch, capsys)
        settings_path = proj / ".claude" / "settings.local.json"
        first = json.loads(settings_path.read_text(encoding="utf-8"))

        run_init(argv, monkeypatch, capsys)
        second = json.loads(settings_path.read_text(encoding="utf-8"))

        assert first == second
        assert len(second["hooks"]["PreToolUse"]) == 1

    def test_existing_policy_without_force_aborts(self, tmp_path, monkeypatch, capsys):
        proj = project(tmp_path, monkeypatch=monkeypatch)
        policy_path = proj / ".claude" / "janus" / "policy.json"
        policy_path.parent.mkdir(parents=True)
        policy_path.write_text('{"Bash": []}')

        code, out = run_init(["init", "--yes", "--project-dir", str(proj)], monkeypatch, capsys)
        assert code == 1
        assert "Aborted" in out
        assert json.loads(policy_path.read_text()) == {"Bash": []}

    def test_scope_flag_selects_the_user_settings_file(self, tmp_path, monkeypatch, capsys):
        home = tmp_path / "home"
        proj = project(tmp_path, home=home, monkeypatch=monkeypatch)
        code, _ = run_init(
            ["init", "--yes", "--scope", "user", "--project-dir", str(proj)],
            monkeypatch,
            capsys,
        )
        assert code == 0
        assert (home / ".claude" / "settings.json").exists()
        assert not (proj / ".claude" / "settings.local.json").exists()


class TestWizardFlow:
    def test_answers_shape_the_policy_and_the_command(self, tmp_path, monkeypatch, capsys):
        proj = project(tmp_path, monkeypatch=monkeypatch)
        # scope=project-local, extra paths, network=open, allow push=y,
        # mode=gate, headless=y, backstop=y
        script = "2\nsecrets/\n3\ny\n\ny\n\n\ny\n"
        code, out = run_init(
            ["init", "--project-dir", str(proj)], monkeypatch, capsys, stdin=script
        )
        assert code == 0, out

        policy = json.loads(
            (proj / ".claude" / "janus" / "policy.json").read_text(encoding="utf-8")
        )
        assert "secrets/" in policy["Read"][0]["conditions"]["file_path"]["pattern"]
        # git push allowed -> no push deny in the policy or the backstop
        assert "git" not in policy["Bash"][0]["conditions"]["command"]["pattern"]

        settings = json.loads(
            (proj / ".claude" / "settings.local.json").read_text(encoding="utf-8")
        )
        command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        assert "--headless" in command
        assert "--mode gate" in command
        assert "Bash(git push:*)" not in settings["permissions"]["deny"]

    def test_declining_at_the_review_screen_writes_nothing(self, tmp_path, monkeypatch, capsys):
        proj = project(tmp_path, monkeypatch=monkeypatch)
        script = "\n\n\n\n\n\n\nn\n"
        code, out = run_init(
            ["init", "--project-dir", str(proj)], monkeypatch, capsys, stdin=script
        )
        assert code == 1
        assert "Nothing was written" in out
        assert not (proj / ".claude" / "janus").exists()

    def test_mcp_servers_produce_a_sidecar_and_config_flag(self, tmp_path, monkeypatch, capsys):
        proj = project(tmp_path, monkeypatch=monkeypatch)
        (proj / ".mcp.json").write_text(json.dumps({"mcpServers": {"research": {}, "tickets": {}}}))
        code, out = run_init(["init", "--yes", "--project-dir", str(proj)], monkeypatch, capsys)
        assert code == 0, out

        sidecar = json.loads(
            (proj / ".claude" / "janus" / "config.json").read_text(encoding="utf-8")
        )
        assert sidecar == {"known_servers": ["research", "tickets"]}

        settings = json.loads(
            (proj / ".claude" / "settings.local.json").read_text(encoding="utf-8")
        )
        assert "--config" in settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]

    def test_no_mcp_servers_means_no_sidecar(self, tmp_path, monkeypatch, capsys):
        proj = project(tmp_path, monkeypatch=monkeypatch)
        run_init(["init", "--yes", "--project-dir", str(proj)], monkeypatch, capsys)
        assert not (proj / ".claude" / "janus" / "config.json").exists()

    def test_foreign_settings_content_survives(self, tmp_path, monkeypatch, capsys):
        proj = project(tmp_path, monkeypatch=monkeypatch)
        settings_path = proj / ".claude" / "settings.local.json"
        settings_path.write_text(json.dumps(settings_fixture("foreign-hooks")))

        code, _ = run_init(["init", "--yes", "--project-dir", str(proj)], monkeypatch, capsys)
        assert code == 0

        after = json.loads(settings_path.read_text(encoding="utf-8"))
        assert after["model"] == "opus"
        assert after["hooks"]["PostToolUse"][0]["hooks"][0]["command"] == "some-other-linter post"
        assert "Bash(rm -rf:*)" in after["permissions"]["deny"]
        backups = list((proj / ".claude").glob("settings.local.json.bak-*"))
        assert len(backups) == 1, "the previous settings file was not backed up"


class TestHookCommandQuoting:
    """The CLI runs this string through a shell. A command the shell mis-parses
    is a hook that never runs — and hook dispatch failure fails OPEN."""

    @staticmethod
    def _command(tmp_path, scope, name="proj"):
        from janus.cli import init as init_module

        proj = tmp_path / name
        proj.mkdir(parents=True, exist_ok=True)
        env = init_module.WizardEnv(
            project_dir=proj, home=tmp_path / "home", hook_executable="janus-hook"
        )
        answers = init_module.WizardAnswers(scope=scope)
        paths = init_module.paths_for_scope(scope, proj, env.home)
        return init_module.build_hook_command(paths, answers, env)

    def test_a_path_with_spaces_survives_shell_splitting(self, tmp_path):
        import shlex

        command = self._command(tmp_path, "user", name="my project")
        tokens = shlex.split(command, posix=os.name != "nt")
        policy = tokens[tokens.index("--policy") + 1]
        assert policy.endswith("policy.json")
        assert Path(policy).name == "policy.json"

    def test_posix_quoting_neutralizes_metacharacters(self, monkeypatch):
        """Exercised on every platform: the POSIX branch is the one that has to
        survive `$(...)`, and CI is otherwise the only place it is checked."""
        from janus.cli import init as init_module

        monkeypatch.setattr(init_module.os, "name", "posix")
        assert init_module._quote("/a b/c") == "'/a b/c'"
        assert init_module._quote("/a$(touch pwned)/c") == "'/a$(touch pwned)/c'"

    def test_windows_refuses_a_path_it_cannot_quote(self, monkeypatch):
        """Better to stop than to emit a command that silently does not run —
        a hook that fails to launch fails open."""
        from janus.cli import init as init_module
        from janus.cli._console import Aborted

        monkeypatch.setattr(init_module.os, "name", "nt")
        assert init_module._quote("C:/a b/c") == '"C:/a b/c"'
        with pytest.raises(Aborted):
            init_module._quote('C:/a"b/c')

    @pytest.mark.skipif(os.name == "nt", reason="POSIX quoting")
    def test_project_scope_keeps_the_variable_expandable(self, tmp_path):
        command = self._command(tmp_path, "project")
        assert "$CLAUDE_PROJECT_DIR/.claude/janus/policy.json" in command
        assert "'$CLAUDE_PROJECT_DIR" not in command, "single quotes block expansion"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX quoting")
    def test_shell_metacharacters_survive_as_a_literal_path(self, tmp_path):
        """A project directory can legally contain `$(...)` on POSIX. It has to
        reach the shim as a path, not as a substitution the shell runs."""
        import shlex

        proj_name = "a b;$(touch pwned)"
        command = self._command(tmp_path, "user", name=proj_name)
        tokens = shlex.split(command)

        assert tokens[0] == "janus-hook"
        policy = tokens[tokens.index("--policy") + 1]
        assert policy == (tmp_path / "home" / ".claude" / "janus" / "policy.json").as_posix()
        # Unquoted, the shell would have executed this before the shim ever ran.
        assert "$(touch pwned)" not in command or "'" in command


class TestVerification:
    def test_default_deployment_passes_its_own_probes(self, tmp_path, monkeypatch, capsys):
        proj = project(tmp_path, monkeypatch=monkeypatch)
        code, out = run_init(["init", "--yes", "--project-dir", str(proj)], monkeypatch, capsys)
        assert code == 0
        assert "FAIL" not in out
        assert "pipe-to-shell download is denied" in out
        assert "ordinary source reads still work" in out

    def test_a_broken_policy_fails_the_probes(self, tmp_path, monkeypatch, capsys):
        """The probes must exercise the written policy, not the wizard's memory
        of what it built."""
        from janus.cli import init as init_module

        proj = project(tmp_path, monkeypatch=monkeypatch)
        original = init_module.build_policy
        monkeypatch.setattr(
            init_module,
            "build_policy",
            lambda answers: {
                tool: ([r for r in rules if r["effect"] == 0] if tool == "Bash" else rules)
                for tool, rules in original(answers).items()
            },
        )
        code, out = run_init(["init", "--yes", "--project-dir", str(proj)], monkeypatch, capsys)
        assert code == 1
        assert "FAIL  pipe-to-shell download is denied" in out
        assert "Some checks failed" in out


class TestLLMAssist:
    def test_skipped_without_the_extra_or_a_key(self, tmp_path, monkeypatch, capsys):
        from janus.cli import init as init_module

        monkeypatch.setattr(
            init_module, "_generator_available", lambda: (False, "OPENAI_API_KEY is not set")
        )
        proj = project(tmp_path, monkeypatch=monkeypatch)
        code, out = run_init(["init", "--yes", "--project-dir", str(proj)], monkeypatch, capsys)
        assert code == 0
        assert "Skipping optional AI-drafted rules" in out

    def test_accepting_replaces_the_blanket_allow(self, monkeypatch):
        """Generated rules land at priority 100, behind the starter's allow@10.
        Appending them would produce rules that can never match, so acceptance
        must swap the blanket allow out."""
        from janus.cli import init as init_module
        from janus.cli._console import Console

        monkeypatch.setattr(init_module, "_generator_available", lambda: (True, ""))
        monkeypatch.setattr(
            init_module,
            "generate_policy",
            lambda *a, **k: {"Bash": [(100, 0, {"command": {"pattern": "^ls"}}, 0)]},
            raising=False,
        )
        import janus.policy.generator as generator

        monkeypatch.setattr(
            generator,
            "generate_policy",
            lambda *a, **k: {"Bash": [(100, 0, {"command": {"pattern": "^ls"}}, 0)]},
        )

        console = Console(io.StringIO("y\nbuild a website\ny\n"), io.StringIO())
        policy = init_module.build_policy(init_module.WizardAnswers())
        merged = init_module.maybe_llm_assist(console, policy)

        effects = [r["effect"] for r in merged["Bash"]]
        assert 1 in effects, "the deny rules were dropped"
        assert not any(r["effect"] == 0 and r["conditions"] == {} for r in merged["Bash"]), (
            "the unconditional allow survived, shadowing every generated rule"
        )
        assert any(r["priority"] == 100 for r in merged["Bash"])

    def test_declining_leaves_the_policy_untouched(self, monkeypatch):
        from janus.cli import init as init_module
        from janus.cli._console import Console

        monkeypatch.setattr(init_module, "_generator_available", lambda: (True, ""))
        console = Console(io.StringIO("n\n"), io.StringIO())
        policy = init_module.build_policy(init_module.WizardAnswers())
        assert init_module.maybe_llm_assist(console, policy) == policy


class TestConsole:
    @staticmethod
    def _console(script: str = "", **kwargs):
        from janus.cli._console import Console

        return Console(io.StringIO(script), io.StringIO(), **kwargs)

    def test_blank_input_takes_the_default(self):
        console = self._console("\n\n")
        assert console.ask_yn("Proceed?", default=True) is True
        assert console.ask_yn("Proceed?", default=False) is False

    def test_yes_no_spellings(self):
        console = self._console("yes\nn\nY\n")
        assert console.ask_yn("a", default=False) is True
        assert console.ask_yn("b", default=True) is False
        assert console.ask_yn("c", default=False) is True

    def test_invalid_answer_reprompts_rather_than_guessing(self):
        console = self._console("maybe\ny\n")
        assert console.ask_yn("Proceed?", default=False) is True

    def test_choice_returns_the_value_not_the_index(self):
        console = self._console("2\n")
        options = [("gate", "Gate mode"), ("policy", "Policy mode")]
        assert console.ask_choice("Mode?", options, default=0) == "policy"

    def test_choice_out_of_range_reprompts(self):
        console = self._console("9\n1\n")
        options = [("gate", "Gate"), ("policy", "Policy")]
        assert console.ask_choice("Mode?", options, default=1) == "gate"

    def test_list_splits_and_trims(self):
        console = self._console("secrets/ , *.key ,,\n")
        assert console.ask_list("Extra paths?") == ["secrets/", "*.key"]

    def test_assume_defaults_never_reads_the_stream(self):
        """--yes must not consume stdin: in CI stdin is often the pipe feeding
        something else entirely."""
        console = self._console("this should not be read\n", assume_defaults=True)
        assert console.ask_yn("Proceed?", default=True) is True
        assert console.ask_text("Paths?", default="x") == "x"
        assert console.ask_choice("Mode?", [("a", "A"), ("b", "B")], default=1) == "b"
        assert console._stdin.read() == "this should not be read\n"

    def test_exhausted_input_aborts(self):
        """A wizard that hits EOF must stop, not silently accept defaults for
        every remaining question."""
        from janus.cli._console import Aborted

        console = self._console("")
        with pytest.raises(Aborted):
            console.ask_yn("Proceed?", default=True)


class TestWriteSettings:
    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / ".claude" / "settings.json"
        assert write_settings(path, {"a": 1}) is None
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}

    def test_existing_file_is_backed_up(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text('{"original": true}', encoding="utf-8")

        backup = write_settings(path, {"replaced": True})

        assert backup is not None and backup.exists()
        assert json.loads(backup.read_text(encoding="utf-8")) == {"original": True}
        assert json.loads(path.read_text(encoding="utf-8")) == {"replaced": True}

    def test_no_temp_files_are_left_behind(self, tmp_path):
        path = tmp_path / "settings.json"
        write_settings(path, {"a": 1})
        assert [p.name for p in tmp_path.iterdir()] == ["settings.json"]

    def test_output_ends_with_a_newline(self, tmp_path):
        path = tmp_path / "settings.json"
        write_settings(path, {"a": 1})
        assert path.read_text(encoding="utf-8").endswith("}\n")
