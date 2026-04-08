"""SkillLoader — 技能元数据扫描与完整内容加载。

扫描 skills/ 目录下所有子文件夹中的 SKILL.md，
解析 YAML 前缀提取元数据（name + description），
缓存元数据列表，按名称按需加载完整内容。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SkillMetadata:
    """Skill 元数据（仅 name + description + path）。"""

    name: str
    description: str
    path: str  # SKILL.md 文件的绝对路径


class SkillLoader:
    """技能加载器：元数据扫描 + 按需完整内容加载。"""

    def __init__(self, skills_dir: Path | str) -> None:
        self._skills_dir = Path(skills_dir)
        self._metadata_cache: list[SkillMetadata] = []
        self._scanned = False

    def scan(self) -> list[SkillMetadata]:
        """扫描 skills/ 目录，解析每个子文件夹中 SKILL.md 的 YAML 前缀。

        Returns:
            解析成功的 SkillMetadata 列表（缓存）。
        """
        self._metadata_cache.clear()

        if not self._skills_dir.is_dir():
            logger.warning("Skills 目录不存在: %s", self._skills_dir)
            self._scanned = True
            return self._metadata_cache

        for child in sorted(self._skills_dir.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            skill_file = child / "SKILL.md"
            if not skill_file.is_file():
                logger.debug("跳过无 SKILL.md 的目录: %s", child.name)
                continue
            meta = self._parse_frontmatter(skill_file)
            if meta:
                self._metadata_cache.append(meta)

        self._scanned = True
        logger.info(
            "Skill 元数据扫描完成: %d 个技能 (%s)",
            len(self._metadata_cache),
            ", ".join(m.name for m in self._metadata_cache),
        )
        return self._metadata_cache

    def get_metadata_list(self) -> list[SkillMetadata]:
        """获取缓存的元数据列表。未扫描则先触发扫描。"""
        if not self._scanned:
            self.scan()
        return list(self._metadata_cache)

    def load_content(self, name: str) -> str | None:
        """按名称加载完整 SKILL.md 内容。

        Args:
            name: Skill 名称（= 子文件夹名）。

        Returns:
            完整文件内容字符串，找不到则返回 None。
        """
        if not self._scanned:
            self.scan()

        for meta in self._metadata_cache:
            if meta.name == name:
                try:
                    return Path(meta.path).read_text(encoding="utf-8")
                except Exception:
                    logger.exception("加载 Skill 内容失败: %s", meta.path)
                    return None

        logger.warning("未找到 Skill: %s", name)
        return None

    @staticmethod
    def _parse_frontmatter(skill_file: Path) -> SkillMetadata | None:
        """解析 SKILL.md 文件的 YAML 前缀，提取 name 和 description。"""
        try:
            text = skill_file.read_text(encoding="utf-8")
        except Exception:
            logger.exception("读取 SKILL.md 失败: %s", skill_file)
            return None

        # 简易 YAML 前缀解析：匹配 --- 包裹的块
        lines = text.split("\n")
        if not lines or lines[0].strip() != "---":
            logger.warning("SKILL.md 缺少 YAML 前缀: %s", skill_file)
            return None

        name = ""
        description = ""
        for line in lines[1:]:
            stripped = line.strip()
            if stripped == "---":
                break
            if stripped.startswith("name:"):
                name = stripped[len("name:"):].strip().strip("'\"")
            elif stripped.startswith("description:"):
                description = stripped[len("description:"):].strip().strip("'\"")

        if not name:
            logger.warning("SKILL.md YAML 缺少 name 字段: %s", skill_file)
            return None

        return SkillMetadata(
            name=name,
            description=description,
            path=str(skill_file.resolve()),
        )
