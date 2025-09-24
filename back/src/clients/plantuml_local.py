# src/clients/plantuml_local.py
from __future__ import annotations
import os, shutil, subprocess

def _has_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def render_plantuml_local(source: str, out: str = "svg", timeout: int = 20):
    """
    Renderiza PlantUML localmente.
    Prioriza el binario `plantuml`. Si no existe, intenta `java -jar $PLANTUML_JAR -pipe`.
    Devuelve (ok: bool, payload: str, err: str|None) donde payload es SVG o PNG (texto/bytes en str).
    """
    fmt = "svg" if (out or "").lower() == "svg" else "png"
    args = []
    use_java = False

    if _has_cmd("plantuml"):
        args = ["plantuml", f"-t{fmt}", "-pipe"]
    else:
        jar = os.environ.get("PLANTUML_JAR", "").strip()
        if not jar or not os.path.exists(jar):
            return False, "", "PlantUML no disponible: instala `plantuml` o define PLANTUML_JAR con la ruta al jar."
        use_java = True
        args = ["java", "-Djava.awt.headless=true", "-jar", jar, f"-t{fmt}", "-pipe"]

    try:
        proc = subprocess.run(
            args,
            input=source.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        if proc.returncode != 0:
            return False, "", f"PlantUML error ({'java' if use_java else 'plantuml'}): {proc.stderr.decode('utf-8', 'ignore')}"
        # SVG llega como texto; PNG llega como bytes: normalizamos a str (base64 no hace falta aquí)
        payload = proc.stdout.decode("utf-8", "ignore") if fmt == "svg" else proc.stdout
        return True, payload, None
    except Exception as e:
        return False, "", f"PlantUML exception: {e}"
