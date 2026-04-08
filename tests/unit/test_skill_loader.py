"""SkillLoader 单元测试。

验证：
- Skill 目录扫描与元数据解析
- 缓存机制
- 按名称加载完整内容
- 各种异常与边界条件
"""

from __future__ import annotations

import textwrap

import pytest

from human_resource.skills.loader import SkillLoader, SkillMetadata


@pytest.fixture()
def skills_dir(tmp_path):
    """创建测试用 skills 目录结构。"""
    # 正常 Skill
    skill_a = tmp_path / "skill_a"
    skill_a.mkdir()
    (skill_a / "SKILL.md").write_text(
        textwrap.dedent("""\
            ---
            name: skill_a
            description: '技能 A 的描述'
            ---
            # Skill A
            详细内容。
        """),
        encoding="utf-8",
    )

    # 第二个 Skill
    skill_b = tmp_path / "skill_b"
    skill_b.mkdir()
    (skill_b / "SKILL.md").write_text(
        textwrap.dedent("""\
            ---
            name: skill_b
            description: "技能 B 的描述"
            ---
            # Skill B
        """),
        encoding="utf-8",
    )

    # 缺少 SKILL.md 的目录（应跳过）
    no_skill = tmp_path / "no_skill"
    no_skill.mkdir()

    # 隐藏目录（应跳过）
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "SKILL.md").write_text("---\nname: hidden\n---\n", encoding="utf-8")

    # 下划线前缀目录（应跳过）
    underscored = tmp_path / "_internal"
    underscored.mkdir()
    (underscored / "SKILL.md").write_text("---\nname: internal\n---\n", encoding="utf-8")

    # 缺少 YAML 前缀的 SKILL.md
    bad_yaml = tmp_path / "bad_yaml"
    bad_yaml.mkdir()
    (bad_yaml / "SKILL.md").write_text("没有 YAML 前缀\n", encoding="utf-8")

    # 缺少 name 字段的 SKILL.md
    no_name = tmp_path / "no_name"
    no_name.mkdir()
    (no_name / "SKILL.md").write_text(
        "---\ndescription: '只有描述'\n---\n", encoding="utf-8",
    )

    return tmp_path


class TestSkillLoaderScan:
    """测试 SkillLoader.scan() 扫描逻辑。"""

    def test_scan_finds_valid_skills(self, skills_dir):
        loader = SkillLoader(skills_dir)
        result = loader.scan()
        names = [m.name for m in result]
        assert "skill_a" in names
        assert "skill_b" in names

    def test_scan_skips_hidden_and_underscored(self, skills_dir):
        loader = SkillLoader(skills_dir)
        result = loader.scan()
        names = [m.name for m in result]
        assert "hidden" not in names
        assert "internal" not in names

    def test_scan_skips_missing_skill_md(self, skills_dir):
        loader = SkillLoader(skills_dir)
        result = loader.scan()
        names = [m.name for m in result]
        assert "no_skill" not in names

    def test_scan_skips_bad_yaml(self, skills_dir):
        loader = SkillLoader(skills_dir)
        result = loader.scan()
        names = [m.name for m in result]
        assert "bad_yaml" not in names

    def test_scan_skips_no_name(self, skills_dir):
        loader = SkillLoader(skills_dir)
        result = loader.scan()
        names = [m.name for m in result]
        assert "no_name" not in names

    def test_scan_empty_dir(self, tmp_path):
        loader = SkillLoader(tmp_path)
        result = loader.scan()
        assert result == []

    def test_scan_nonexistent_dir(self, tmp_path):
        loader = SkillLoader(tmp_path / "nonexistent")
        result = loader.scan()
        assert result == []

    def test_scan_extracts_description(self, skills_dir):
        loader = SkillLoader(skills_dir)
        result = loader.scan()
        meta_a = next(m for m in result if m.name == "skill_a")
        assert meta_a.description == "技能 A 的描述"


class TestSkillLoaderCache:
    """测试元数据缓存机制。"""

    def test_get_metadata_list_triggers_scan(self, skills_dir):
        loader = SkillLoader(skills_dir)
        # 未手动 scan，get_metadata_list 应自动触发
        result = loader.get_metadata_list()
        assert len(result) >= 2

    def test_get_metadata_list_returns_copy(self, skills_dir):
        loader = SkillLoader(skills_dir)
        loader.scan()
        list1 = loader.get_metadata_list()
        list2 = loader.get_metadata_list()
        assert list1 == list2
        assert list1 is not list2  # 返回副本


class TestSkillLoaderLoadContent:
    """测试按名称加载完整 SKILL.md 内容。"""

    def test_load_existing_skill(self, skills_dir):
        loader = SkillLoader(skills_dir)
        loader.scan()
        content = loader.load_content("skill_a")
        assert content is not None
        assert "# Skill A" in content
        assert "name: skill_a" in content

    def test_load_nonexistent_skill(self, skills_dir):
        loader = SkillLoader(skills_dir)
        loader.scan()
        content = loader.load_content("nonexistent")
        assert content is None

    def test_load_triggers_scan_if_needed(self, skills_dir):
        loader = SkillLoader(skills_dir)
        content = loader.load_content("skill_b")
        assert content is not None
        assert "# Skill B" in content


class TestSkillMetadata:
    """测试 SkillMetadata 数据模型。"""

    def test_dataclass_fields(self):
        meta = SkillMetadata(name="test", description="desc", path="/tmp/SKILL.md")
        assert meta.name == "test"
        assert meta.description == "desc"
        assert meta.path == "/tmp/SKILL.md"
