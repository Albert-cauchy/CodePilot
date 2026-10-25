import json

import pytest

from codepilot.tools.think import create_think_tool


def test_think_creates_then_updates_only_its_own_plan(tmp_path):
    think = create_think_tool(tmp_path)
    result = json.loads(
        think.invoke(
            {
                "steps": [
                    {"id": "1", "description": "定位错误", "status": "in_progress"},
                    {"id": "2", "description": "修复并验证", "status": "pending"},
                ]
            }
        )
    )
    assert result["operation"] == "write"
    assert (tmp_path / "plan.md").read_text() == result["plan"]
    updated = json.loads(
        think.invoke(
            {
                "steps": [
                    {"id": "1", "description": "定位错误", "status": "completed"},
                    {"id": "2", "description": "修复并验证", "status": "in_progress"},
                ]
            }
        )
    )
    assert updated["operation"] == "edit"
    assert "completed" in updated["plan"]
    assert (tmp_path / "plan.md").read_text() == updated["plan"]
    with pytest.raises(ValueError):
        think.invoke({"steps": [], "path": str(tmp_path / "other.md")})
    assert (tmp_path / "plan.md").read_text() == updated["plan"]
    assert not (tmp_path / "other.md").exists()
