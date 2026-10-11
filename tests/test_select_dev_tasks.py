import pytest

from scripts.select_dev_tasks import select_development_tasks


def test_selection_covers_repos_and_tracks_proportions():
    rows = [
        {"repo": repo, "instance_id": f"{repo}-{index}"}
        for repo, count in (("large", 12), ("medium", 6), ("small", 2))
        for index in range(count)
    ]

    selected, counts = select_development_tasks(rows, sample_size=10, seed=42)

    assert len(selected) == len(set(selected)) == 10
    assert {repo: item["selected"] for repo, item in counts.items()} == {
        "large": 6,
        "medium": 3,
        "small": 1,
    }
    assert {repo: item["verified"] for repo, item in counts.items()} == {
        "large": 12,
        "medium": 6,
        "small": 2,
    }
    assert selected == select_development_tasks(reversed(rows), sample_size=10, seed=42)[0]


def test_selection_requires_one_slot_per_repository():
    rows = [{"repo": str(index), "instance_id": str(index)} for index in range(3)]

    with pytest.raises(ValueError, match="covering every repository"):
        select_development_tasks(rows, sample_size=2)
