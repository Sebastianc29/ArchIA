import React from "react";
import { AppBar, Toolbar, Typography, IconButton, Box } from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import "../styles/header.css";

/** Header global único.
 *  Notifica al chat con un CustomEvent para abrir/cerrar el drawer. */
export default function Header() {
  const toggleDrawer = () => {
    window.dispatchEvent(new CustomEvent("arquia-toggle-drawer"));
  };

  return (
    <AppBar
      position="static"
      sx={{
        background: "#111",
        boxShadow: "0 1px 0 rgba(255,255,255,0.06)",
      }}
    >
      <Toolbar sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <IconButton edge="start" aria-label="menu" onClick={toggleDrawer} sx={{ color: "#fff" }}>
          <MenuIcon />
        </IconButton>

        <Box sx={{ display: "flex", alignItems: "baseline", gap: 2 }}>
          <Typography variant="h6" sx={{ color: "#fff", fontWeight: 700 }}>
            ArquIA
          </Typography>
          <Typography variant="h6" sx={{ color: "#d3d3d3", fontWeight: 500 }}>
            Chat
          </Typography>
        </Box>
      </Toolbar>
    </AppBar>
  );
}
