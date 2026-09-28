"""Config loading, merging and persona resolution."""

from __future__ import annotations

import pytest

from ohbot_kit import config as config_mod


def write(tmp_path, name, text):  # type: ignore[no-untyped-def]
    path = tmp_path / name
    path.write_text(text)
    return str(path)


class TestPrecedence:
    def test_local_overrides_shared(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """config.local.yaml is git-ignored and holds machine-specific values,
        so it has to win over the committed defaults."""
        base = write(tmp_path, "config.yaml", "llm:\n  model: phi4-mini\n  temperature: 0.7\n")
        local = write(tmp_path, "config.local.yaml", "llm:\n  model: qwen3.6:35b\n")
        cfg = config_mod.load(base, local, warn=False)

        assert cfg.get("llm.model") == "qwen3.6:35b"
        # A partial override must not wipe its siblings.
        assert cfg.get("llm.temperature") == 0.7

    def test_merge_is_deep(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        base = write(tmp_path, "config.yaml", "audio:\n  input_device: mic\n  output_device: spk\n")
        local = write(tmp_path, "config.local.yaml", "audio:\n  output_device: headset\n")
        cfg = config_mod.load(base, local, warn=False)
        assert cfg.get("audio.input_device") == "mic"
        assert cfg.get("audio.output_device") == "headset"

    def test_missing_files_are_fine(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A participant with no config at all must still be able to run."""
        cfg = config_mod.load(str(tmp_path / "nope.yaml"), str(tmp_path / "nope2.yaml"), warn=False)
        assert cfg.sources == []
        assert cfg.get("llm.model", "fallback") == "fallback"


class TestGet:
    def test_dotted_path(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = write(tmp_path, "c.yaml", "robot:\n  eye_colours:\n    thinking: [10, 5, 0]\n")
        cfg = config_mod.load(path, None, warn=False)
        assert cfg.get("robot.eye_colours.thinking") == [10, 5, 0]

    def test_null_means_use_the_default(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """`input_device: null` in the shipped config means "system default",
        so an explicit null must behave like a missing key."""
        path = write(tmp_path, "c.yaml", "audio:\n  input_device: null\n")
        cfg = config_mod.load(path, None, warn=False)
        assert cfg.get("audio.input_device", "fallback") == "fallback"

    def test_missing_path_returns_default(self) -> None:
        assert config_mod.Config({}).get("a.b.c", "d") == "d"

    def test_descending_through_a_non_dict(self) -> None:
        cfg = config_mod.Config({"a": "string"})
        assert cfg.get("a.b", "default") == "default"


class TestPersonas:
    def _cfg(self, tmp_path):  # type: ignore[no-untyped-def]
        path = write(
            tmp_path,
            "c.yaml",
            "persona: friendly\n"
            "personas:\n"
            "  friendly:\n    voice: af_heart\n    system_prompt: be nice\n"
            "  pirate:\n    voice: am_michael\n    system_prompt: arr\n",
        )
        return config_mod.load(path, None, warn=False)

    def test_default_persona(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        assert self._cfg(tmp_path).persona()["voice"] == "af_heart"

    def test_named_persona(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        assert self._cfg(tmp_path).persona("pirate")["voice"] == "am_michael"

    def test_unknown_persona_lists_the_valid_ones(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(KeyError, match="friendly, pirate"):
            self._cfg(tmp_path).persona("wizard")


class TestValidation:
    def test_unknown_section_warns_but_still_loads(self, tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
        """A typo shouldn't kill a demo, but it shouldn't be silent either --
        an ignored setting looks exactly like one that did nothing."""
        path = write(tmp_path, "c.yaml", "llm:\n  model: x\ntypoed_section:\n  a: 1\n")
        cfg = config_mod.load(path, None, warn=True)
        assert "typoed_section" in capsys.readouterr().err
        assert cfg.get("llm.model") == "x"

    def test_malformed_yaml_names_the_file(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = write(tmp_path, "c.yaml", "llm:\n  model: [unclosed\n")
        with pytest.raises(RuntimeError, match="c.yaml"):
            config_mod.load(path, None, warn=False)

    def test_non_mapping_rejected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = write(tmp_path, "c.yaml", "- just\n- a\n- list\n")
        with pytest.raises(RuntimeError, match="mapping"):
            config_mod.load(path, None, warn=False)


class TestUpdateLocal:
    def test_preserves_leading_comment_block_and_siblings(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """config.local.yaml's machine-specific notes live in a leading
        comment block since PyYAML drops comments on a full rewrite -- a
        write-back must not lose them, and a partial update must not wipe
        sibling keys like a plain re-dump would risk."""
        local = write(
            tmp_path,
            "config.local.yaml",
            "# Machine-specific notes.\n"
            "# More notes.\n"
            "\n"
            "audio:\n"
            "  input_device: Plantronics\n"
            "  output_device: Plantronics\n",
        )
        config_mod.update_local({"audio": {"output_device": "MacBook Pro Speakers"}}, local)

        text = open(local).read()
        assert "# Machine-specific notes." in text
        assert "# More notes." in text

        cfg = config_mod.load(None, local, warn=False)
        assert cfg.get("audio.output_device") == "MacBook Pro Speakers"
        assert cfg.get("audio.input_device") == "Plantronics"

    def test_creates_file_when_missing(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        local = str(tmp_path / "config.local.yaml")
        config_mod.update_local({"audio": {"output_device": "BlackHole 16ch"}}, local)

        cfg = config_mod.load(None, local, warn=False)
        assert cfg.get("audio.output_device") == "BlackHole 16ch"

    def test_resetting_to_none_writes_null(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Picking "System default" in the UI should behave exactly like the
        shipped config's `output_device: null`, not like an unset key."""
        local = write(tmp_path, "config.local.yaml", "audio:\n  output_device: Plantronics\n")
        config_mod.update_local({"audio": {"output_device": None}}, local)

        assert "output_device: null" in open(local).read()
        cfg = config_mod.load(None, local, warn=False)
        assert cfg.get("audio.output_device", "fallback") == "fallback"
