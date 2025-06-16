from schemas.supreme_court import SupremeCourtCase, CaseChunk

case = SupremeCourtCase(
    id="410 U.S. 113",
    docket_number="70-18",
    name="Roe v. Wade",
    # ... other fields
)

chunk = CaseChunk(
    case_id=case.id,
    chunk_index=0,
    total_chunks=10,
    text="The Constitution does not explicitly mention any right of privacy...",
    metadata={"topic": "Right to Privacy"}
)