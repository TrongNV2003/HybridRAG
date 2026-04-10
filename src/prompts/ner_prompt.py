EXTRACT_SYSTEM_PROMPT = """You are an AI assistant expert in knowledge extraction for DBPedia. Your task is to extract entities and their relationships from text and return them as valid JSON. Follow the DBPedia ontology standards (Person, Place, Organisation, etc.) strictly."""

EXTRACT_PROMPT_TEMPLATE = (
    "### Role:\n"
    "You are an expert in Knowledge Graph extraction for DBPedia.\n"
    "\n"
    "### Instruction: \n"
    "- Analyze the following Vietnamese text and extract entities and relationships between them following DBPedia standards.\n"
    "1.  **Entities:**\n"
    "- Each entity must have an 'id' (name), 'entity_type', and 'entity_role'.\n"
    "- Use standard DBPedia types: Person, Place, Organisation, Work, Event, Species, etc.\n"
    "- 'entity_role' should describe the entity's specific role in the context (e.g., 'Nhà cách mạng', 'Thủ đô').\n"
    "2.  **Relationships:**\n"
    "    - Use meaningful relationship types like: 'sinh_tại', 'là_thủ_đô_của', 'thành_viên_của', 'tác_giả_của', 'thuộc_tỉnh'.\n"
    '    - `source` and `target` MUST match exactly the `id` field of the extracted nodes.\n'
    "\n"
    "## Example output (valid JSON):\n"
    '{"nodes": [{"id": "Hồ Chí Minh", "entity_type": "Person", "entity_role": "Nhà cách mạng"}],\n'
    '"relationships": [{"source": "Hồ Chí Minh", "target": "Nghệ An", "relationship_type": "sinh_tại"}]}\n'
    "\n"
    "### Execute with the following input (Vietnamese Text)\n"
    "<input>\n"
    "{{ text }}\n"
    "</input>\n"
)

EXTRACT_SCHEMA = {
    "name": "entities_extraction",
    "schema": {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Name of the entity being extracted."
                        },
                        "entity_type": {
                            "type": "string",
                            "description": "Type of the entity (e.g., Person, Place, Event, Organization)."
                        },
                        "entity_role": {
                            "type": "string",
                            "description": "Role or function of the entity (e.g., 'Queen of England'). If not applicable, leave as an empty string ('')."
                        },
                    },
                    "required": ["id", "entity_type", "entity_role"],
                    "additionalProperties": False
                }
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Name of the source entity, must match exactly the `id` field of the extracted nodes."
                        },
                        "target": {
                            "type": "string",
                            "description": "Name of the target entity, must match exactly the `id` field of the extracted nodes."
                        },
                        "relationship_type": {
                            "type": "string",
                            "description": "The type of relationship between two entities."
                        }
                    },
                    "required": ["source", "target", "relationship_type"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["nodes", "relationships"],
        "additionalProperties": False
    },
    "strict": True,
}
