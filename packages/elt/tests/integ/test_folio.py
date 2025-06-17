import asyncio
from folio import FOLIO

folio = FOLIO()

async def search_example(text: str):
    for result in await folio.parallel_search_by_llm(
        text,
        search_sets=[
            folio.get_areas_of_law(max_depth=1),
            folio.get_player_actors(max_depth=2),
        ],
    ):
        print(result)

def test_search_example():
    # Search for a class by label
    result, score = folio.search_by_label("SCOTUS")[0]
    print(f"{result.iri} (Score: {score})")
    print(f"Preferred Label: {result.preferred_label}")
    print(f"Synonyms: {result.alternative_labels}")
    print(f"Parents: {[folio[c].label for c in result.sub_class_of]}")
    # print(result.__class__)
    print(result.to_json())
    # print(result.to_jsonld())

def test_properties():
    properties = folio.get_all_properties()
    print(f"Number of object properties: {len(properties)}")

    # Get properties by label
    drafted_properties = folio.get_properties_by_label("folio:drafted")
    print(drafted_properties)
    for prop in drafted_properties:
        print(f"Property: {prop.label}")
        print(f"Domain: {[folio[d].label for d in prop.domain if folio[d]]}")
        print(f"Range: {[folio[r].label for r in prop.range if folio[r]]}")

def test_case():
    # Find connections between entities
    connections = folio.find_connections(
        subject_class="https://folio.openlegalstandard.org/RFE94c038Ce43B892dbECa17",  # SCOTUS
        property_name="folio:drafted"
    )
    for subject, property_obj, object_class in connections:
        print(f"{subject.label} {property_obj.label} {object_class.label}")
    
def test_actor():
    # Find connections between entities
    connections = folio.find_connections(
        subject_class="https://folio.openlegalstandard.org/R8CdMpOM0RmyrgCCvbpiLS0",  # Actor/Player
        property_name="folio:drafted"
    )
    for subject, property_obj, object_class in connections:
        print(f"{subject.label} {property_obj.label} {object_class.label}")