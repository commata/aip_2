from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from dogfight.ai.bt_rule_manager import RULE_XML_NAME, activate_rule_xml


class BTRuleManagerTests(unittest.TestCase):
    def test_aliases_are_active_only_inside_context(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            source = root / "official.xml"
            source.write_text("<root>official</root>", encoding="utf-8")
            existing = root / RULE_XML_NAME
            existing.write_text("<root>previous</root>", encoding="utf-8")
            alias = root / "Rule_DCS_GDCC_0815.xml"

            with activate_rule_xml(
                source,
                root,
                aliases=[alias.name],
            ):
                self.assertEqual(existing.read_bytes(), source.read_bytes())
                self.assertEqual(alias.read_bytes(), source.read_bytes())

            self.assertEqual(
                existing.read_text(encoding="utf-8"),
                "<root>previous</root>",
            )
            self.assertFalse(alias.exists())

    def test_rejects_alias_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            source = root / "official.xml"
            source.write_text("<root />", encoding="utf-8")

            with self.assertRaises(ValueError):
                with activate_rule_xml(source, root, aliases=["../outside.xml"]):
                    pass


if __name__ == "__main__":
    unittest.main()
