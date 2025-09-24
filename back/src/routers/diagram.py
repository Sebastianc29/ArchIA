from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from ..clients.kroki_client import render_kroki

router = APIRouter(tags=["diagram"])

class DiagramReq(BaseModel):
    diagram_type: str = Field(pattern="^(mermaid|plantuml|graphviz|c4plantuml|erd|vega|vegalite|svgbob|structurizr)$")
    output_format: str = Field(pattern="^(svg|png|pdf|txt)$", default="svg")
    source: str = Field(min_length=3, max_length=20000)

@router.post("/diagram/render")
def render_diagram(req: DiagramReq):
    try:
        blob, ctype = render_kroki(req.diagram_type, req.source, req.output_format)
        return Response(content=blob, media_type=ctype)
    except Exception as e:
        # Fallback: si es mermaid, devolvemos el código para render local en el front
        if req.diagram_type == "mermaid":
            return {"fallback": True, "syntax": "mermaid", "source": req.source, "error": str(e)}
        raise HTTPException(status_code=502, detail=str(e))
