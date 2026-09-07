from __future__ import annotations

import re
from pathlib import Path

from oem_knowledge.markdown.frontmatter import parse_frontmatter


class ConceptEvolutionEngine:
    def __init__(self, engine):
        self.engine = engine

    def evolve_concept(self, concept_id: str, project: str | None = None) -> dict:
        """Consolidate evidence, deduplicate learnings, and rewrite concept summary."""
        concepts_dir = self.engine._concepts_dir(project)
        concept_file = concepts_dir / f"{concept_id}.md"

        if not concept_file.exists():
            return {"status": "error", "message": f"Concept file {concept_id}.md not found."}

        content = concept_file.read_text(encoding="utf-8")

        parsed = parse_frontmatter(content, source_path=str(concept_file))
        if not parsed.metadata:
            return {"status": "error", "message": "Invalid frontmatter structure."}

        body = parsed.body
        # Find the closing delimiter to extract the original header bytes.
        closing = re.search(r"\n(---|\.\.\.)\s*\n", content)
        header = content[: closing.end()] if closing else ""
        if not header:
            return {"status": "error", "message": "Invalid frontmatter structure."}

        # Parse learnings list
        learnings_section = re.search(r"## Learnings.*$", body, re.DOTALL)
        learnings = []
        if learnings_section:
            for line in learnings_section.group(0).splitlines():
                if line.strip().startswith("-"):
                    item = re.sub(r"^-\s*(\*\*[^*]+\*\*:\s*)?", "", line.strip())
                    if item:
                        learnings.append(item.strip())

        # Deduplicate learnings and extract key sentences
        unique_learnings = []
        seen = set()
        for item in learnings:
            item_lower = item.lower().strip(".")
            if item_lower not in seen:
                seen.add(item_lower)
                unique_learnings.append(item)

        # Rewrite summary based on learnings
        if unique_learnings:
            bullet_points = "\n".join(f"- {item}" for item in unique_learnings)
            new_body = f"# {concept_id.replace('_', ' ').title()}\n\nThis concept has evolved with consolidated evidence.\n\n## Learnings\n{bullet_points}\n"
        else:
            new_body = body

        new_content = header + new_body
        self.engine.materialization._safe_write_concept_file(concept_file, new_content, project)

        return {
            "status": "success",
            "message": f"Concept {concept_id} evolved and consolidated.",
            "learnings_count": len(unique_learnings)
        }

    def propose_merges(
        self,
        similarity_threshold: float = 0.85,
        project: str | None = None,
        *,
        lock: bool = True,
    ) -> list[dict]:
        """Propose merging concepts with highly similar canonical names or aliases."""
        registry = self.engine.state._load_registry(project, lock=lock)
        cids = list(registry.keys())
        proposals = []
        
        import difflib
        
        def similarity(s1: str, s2: str) -> float:
            return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
            
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                cid_a = cids[i]
                cid_b = cids[j]
                data_a = registry[cid_a]
                data_b = registry[cid_b]
                
                name_a = data_a.get("canonical_name", "")
                name_b = data_b.get("canonical_name", "")
                
                sim = similarity(name_a, name_b)
                
                aliases_a = data_a.get("aliases", [])
                aliases_b = data_b.get("aliases", [])
                
                max_alias_sim = 0.0
                for a in aliases_a:
                    max_alias_sim = max(max_alias_sim, similarity(a, name_b))
                for b in aliases_b:
                    max_alias_sim = max(max_alias_sim, similarity(b, name_a))
                for a in aliases_a:
                    for b in aliases_b:
                        max_alias_sim = max(max_alias_sim, similarity(a, b))
                        
                best_sim = max(sim, max_alias_sim)
                if best_sim >= similarity_threshold:
                    status_rank = {"canonical": 4, "validated": 3, "emerging": 2, "candidate": 1}
                    rank_a = (status_rank.get(data_a.get("status"), 0), data_a.get("evidence_count", 0))
                    rank_b = (status_rank.get(data_b.get("status"), 0), data_b.get("evidence_count", 0))
                    
                    if rank_a >= rank_b:
                        primary, secondary = cid_a, cid_b
                    else:
                        primary, secondary = cid_b, cid_a
                        
                    proposals.append({
                        "primary_id": primary,
                        "secondary_id": secondary,
                        "primary_name": registry[primary].get("canonical_name", primary),
                        "secondary_name": registry[secondary].get("canonical_name", secondary),
                        "similarity": round(best_sim, 4),
                        "reason": f"High naming similarity ({round(best_sim * 100)}%) between '{registry[primary].get('canonical_name')}' and '{registry[secondary].get('canonical_name')}'"
                    })
                    
        return proposals


class ContradictionDetector:
    def __init__(self, engine):
        self.engine = engine
        self.dense_search = self.engine.search

        # Hardcoded architectural contradiction rule pairs (lowercased)
        self.conflict_rules = [
            (r"rest\b", r"grpc\b", "REST vs gRPC protocol conflict"),
            (r"postgresql\b", r"mysql\b", "PostgreSQL vs MySQL database selection conflict"),
            (r"tabs\b", r"spaces\b", "Tabs vs Spaces formatting conflict"),
            (r"monolith\b", r"microservice\b", "Monolithic vs Microservice architecture conflict"),
            (r"sync\b", r"async\b", "Synchronous vs Asynchronous flow conflict"),
        ]

    def detect_contradictions(self, project: str | None = None) -> list[dict]:
        """Scan all concepts and identify architectural or semantic contradictions."""
        registry = self.engine.state._load_registry(project)
        concepts_dir = self.engine._concepts_dir(project)
        cids = list(registry.keys())

        docs = {}
        for cid in cids:
            wiki_file = concepts_dir / f"{cid}.md"
            if wiki_file.exists():
                docs[cid] = wiki_file.read_text(encoding="utf-8").lower()

        contradictions = []

        # 1. Rule-based static contradiction scanning
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                cid_a = cids[i]
                cid_b = cids[j]
                content_a = docs.get(cid_a, "")
                content_b = docs.get(cid_b, "")

                if not content_a or not content_b:
                    continue

                for pattern_a, pattern_b, desc in self.conflict_rules:
                    if (re.search(pattern_a, content_a) and re.search(pattern_b, content_b)) or \
                       (re.search(pattern_b, content_a) and re.search(pattern_a, content_b)):
                        contradictions.append({
                            "concept_a": cid_a,
                            "concept_b": cid_b,
                            "name_a": registry[cid_a].get("canonical_name", ""),
                            "name_b": registry[cid_b].get("canonical_name", ""),
                            "type": "architectural_conflict",
                            "description": desc
                        })

        return contradictions
