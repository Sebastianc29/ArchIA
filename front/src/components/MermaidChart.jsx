import React, { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';
import "../styles/mermaidChart.css";

const MermaidChart = ({ chart }) => {
  const mermaidRef = useRef(null);
  const [rendered, setRendered] = useState(false);

  // Mantener un ID estable por instancia del componente
  const chartIdRef = useRef(`mermaid-${Math.random().toString(36).substr(2, 9)}`);

  useEffect(() => {
    // Inicializar mermaid solo una vez
    if (!window.mermaidInitialized) {
      mermaid.initialize({
        startOnLoad: false,
        theme: 'dark',
        securityLevel: 'loose',
        fontFamily: 'monospace',
        fontSize: 16,
        themeVariables: {
          // Tus colores y estilos originales
          primaryTextColor: '#ffffff',
          primaryColor: '#434857',
          primaryBorderColor: '#ffffff',
          lineColor: '#d3d3d3',
          secondaryColor: '#2a3052',
          tertiaryColor: '#1a1a2e',
          nodeBorder: '#ffffff',
          clusterBkg: '#23283d',
          clusterBorder: '#ffffff',
          defaultLinkColor: '#d3d3d3',
          titleColor: '#ffffff',
          edgeLabelBackground: '#23283d',
          nodeTextColor: '#1a1a1a'
        }
      });
      window.mermaidInitialized = true;
    }

    // Si no hay diagrama, limpiamos y salimos
    if (!chart || !chart.trim()) {
      setRendered(false);
      if (mermaidRef.current) {
        mermaidRef.current.innerHTML = "";
      }
      return;
    }

    let cancelled = false;

    // Pequeño delay para que el DOM esté listo
    const renderTimer = setTimeout(async () => {
      if (!mermaidRef.current) return;

      // 1) Validar sintaxis ANTES de intentar renderizar
      try {
        mermaid.parse(chart);
      } catch (err) {
        console.warn("Mermaid syntax error. Diagram will not be rendered.", err);
        // No mostramos nada en pantalla, solo limpiamos el contenedor
        if (mermaidRef.current) {
          mermaidRef.current.innerHTML = "";
        }
        setRendered(false);
        return;
      }

      // 2) Si la sintaxis es válida, renderizamos
      try {
        const { svg } = await mermaid.render(chartIdRef.current, chart);
        if (!cancelled && mermaidRef.current) {
          mermaidRef.current.innerHTML = svg;
          setRendered(true);
        }
      } catch (err) {
        console.warn("Failed to render mermaid chart.", err);
        if (mermaidRef.current) {
          mermaidRef.current.innerHTML = "";
        }
        setRendered(false);
      }
    }, 100);

    return () => {
      cancelled = true;
      clearTimeout(renderTimer);
    };
  }, [chart]);

  return (
    <div className="mermaid-container" style={{ minHeight: '100px' }}>
      <div
        ref={mermaidRef}
        className="mermaid-chart"
        data-processed={rendered.toString()}
      ></div>
    </div>
  );
};

export default MermaidChart;
