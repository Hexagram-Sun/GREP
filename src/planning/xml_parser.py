import re
from typing import List, Dict
from src.planning.oracle_discovery import Action

class PlanParser:
    @staticmethod
    def to_xml(plan: List[Action]) -> str:
        xml_str = "<plan>\n"
        for action in plan:
            idx = "1" if "bm25" in action.tool_name.lower() else "0"
            xml_str += f'    <action retriever_index="{idx}">{action.query_text}</action>\n'
        xml_str += "</plan>"
        return xml_str

    @staticmethod
    def parse_xml(xml_string: str) -> List[Dict]:
        actions = []
        pattern = r'<action retriever_index="(\d+)">\s*(.+?)\s*</action>'
        
        matches = re.findall(pattern, xml_string, re.DOTALL)
        
        for idx, query in matches:
            tool = "bm25" if idx == "1" else "dense"
            actions.append({
                "tool_name": tool,
                "query_text": query.strip()
            })
        
        return actions

if __name__ == "__main__":
    sample_plan = [
        Action(tool_name="bm25", query_text="taxonomic rank of [ENT_1]"),
        Action(tool_name="dense", query_text="characteristics of [ENT_2]")
    ]
    xml = PlanParser.to_xml(sample_plan)
    print("Generated XML:\n", xml)
    
    parsed = PlanParser.parse_xml(xml)
    print("\nParsed:\n", parsed)