import React, { useState, useRef, useEffect } from "react";
import MermaidChart from "./MermaidChart";
import {
  Box, Paper, List, ListItem, TextField, Button, Typography,
  Dialog, DialogTitle, DialogContent, ListItemText, IconButton,
  Badge, Chip, Stack, Tooltip, Divider, CircularProgress
} from "@mui/material";
import AttachFileIcon from "@mui/icons-material/AttachFile";
import CloseIcon from "@mui/icons-material/Close";
import ImageIcon from "@mui/icons-material/Image";
import ThumbUpIcon from "@mui/icons-material/ThumbUp";
import ThumbDownIcon from "@mui/icons-material/ThumbDown";
import MoreHorizIcon from "@mui/icons-material/MoreHoriz";
import ScienceIcon from "@mui/icons-material/Science";

import "../styles/chat.css";

const uuid = () => (crypto?.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random()));

/** ——— Helpers de UI/analítica del turno ——— **/

// Saca un mini “conteo por rol” (deduplicado por nombre)
const summarizeRoles = (internal = []) => {
  const counts = new Map();
  const order = ["supervisor_prompt","supervisor_decision","researcher","evaluator","creator","asr"];
  internal.forEach((m) => {
    const k = (m.name || m.role || "other").toLowerCase();
    counts.set(k, (counts.get(k) || 0) + 1);
  });
  const ordered = order.filter(k => counts.has(k)).map(k => [k, counts.get(k)]);
  const rest = [...counts.entries()].filter(([k]) => !order.includes(k));
  return [...ordered, ...rest];
};

// Encuentra bloque SOURCES: que devuelve local_RAG y lo parsea
const extractRagSources = (internal = []) => {
  const sources = [];
  for (const m of internal) {
    const text = String(m.content || "");
    const idx = text.indexOf("\nSOURCES:\n");
    if (idx !== -1) {
      const lines = text.substring(idx + "\nSOURCES:\n".length).split("\n");
      for (const ln of lines) {
        const t = ln.trim();
        if (t.startsWith("- ")) sources.push(t.slice(2));
      }
    }
  }
  return sources;
};

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [openDialog, setOpenDialog] = useState(false);
  const [selectedInternalMessages, setSelectedInternalMessages] = useState([]);
  const [selectedRagSources, setSelectedRagSources] = useState([]);
  const [attachedImages, setAttachedImages] = useState([]);
  const [sessionId, setSessionId] = useState("");
  const fileInputRef = useRef(null);
  const requestSeq = useRef(0);
  const [ratedMessages, setRatedMessages] = useState(new Set());

  useEffect(() => {
    const newSessionId = uuid();
    setSessionId(newSessionId);
    console.log(`New session created with ID: ${newSessionId}`);
  }, []);

  const sendMessage = async () => {
    if (!input.trim() && attachedImages.length === 0) return;

    const userId = uuid();
    const pendingId = "pending-" + uuid();
    const currentSeq = ++requestSeq.current;

    // pinta turno
    setMessages(prev => ([
      ...prev,
      { id: userId, sender: "usuario", text: input, images: attachedImages.map(img => img.preview) },
      { id: pendingId, sender: "respuesta", text: "", pending: true }
    ]));

    // payload
    const formData = new FormData();
    formData.append("message", input);
    formData.append("session_id", sessionId);
    attachedImages.forEach((img, index) => formData.append(`image${index + 1}`, img.file));

    // limpia UI
    setInput(""); setAttachedImages([]); if (fileInputRef.current) fileInputRef.current.value = "";

    try {
      const resp = await fetch("http://localhost:8000/message", { method: "POST", body: formData });
      const data = await resp.json();
      if (currentSeq !== requestSeq.current) return;

      setMessages(prev =>
        prev.map(m =>
          m.id === pendingId
            ? {
                ...m,
                pending: false,
                text: data.endMessage,
                internal_messages: Array.isArray(data.messages) ? data.messages : [],
                mermaidCode: data.mermaidCode,
                session_id: data.session_id,
                message_id: data.message_id
              }
            : m
        )
      );
    } catch (e) {
      console.error("Error:", e);
      setMessages(prev =>
        prev.map(m =>
          m.id === pendingId
            ? { ...m, pending: false, text: "⚠️ Ocurrió un error al obtener la respuesta." }
            : m
        )
      );
    }
  };

  const handleThumbClick = (session_id, message_id, thumbs_up, thumbs_down) => {
    const formdata = new FormData();
    formdata.append("session_id", session_id);
    formdata.append("message_id", message_id);
    formdata.append("thumbs_up", thumbs_up);
    formdata.append("thumbs_down", thumbs_down);
    fetch("http://localhost:8000/feedback", { method: "POST", body: formdata });
  };

  const handleRating = (sid, mid, isThumbsUp) => {
    const key = `${sid}-${mid}`;
    if (ratedMessages.has(key)) return;
    const next = new Set(ratedMessages); next.add(key); setRatedMessages(next);
    handleThumbClick(sid, mid, isThumbsUp ? 1 : 0, isThumbsUp ? 0 : 1);
  };

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files);
    if (files.length + attachedImages.length > 2) { alert("Solo puedes adjuntar hasta 2 imágenes."); return; }
    const newImages = files.map(file => ({ file, preview: URL.createObjectURL(file), name: file.name }));
    setAttachedImages(prev => [...prev, ...newImages]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };
  const removeImage = (i) => setAttachedImages(prev => prev.filter((_, idx) => idx !== i));

  // ——— UI helpers ———
  const bubbleSx = (isUser) => ({
    width: "100%",
    borderRadius: 12,
    padding: "14px 16px",
    background: isUser ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.10)",
    boxShadow: "0 8px 22px rgba(0,0,0,0.28)",
    position: "relative"
  });
  const headerSx = { display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1 };
  const headerTitle = (msg) => msg.sender === "usuario" ? "Tú" : (msg.pending ? "Asistente (pensando…)" : "Asistente");

  return (
    <Box className="chat-container">
      <Paper className="messages-box">
        <List sx={{ px: 2, py: 1 }}>
          {messages.map((msg) => {
            const isUser = msg.sender === "usuario";
            const roles = !isUser ? summarizeRoles(msg.internal_messages) : [];
            const ragSources = !isUser ? extractRagSources(msg.internal_messages) : [];
            const ratedKey = `${msg.session_id}-${msg.message_id}`;

            return (
              <ListItem key={msg.id} disableGutters sx={{ mb: 2 }}>
                <Box sx={bubbleSx(isUser)}>
                  {/* Header */}
                  <Box sx={headerSx}>
                    <Typography variant="subtitle2" sx={{ color: "#E8EAF6", fontWeight: 600 }}>
                      {headerTitle(msg)}
                    </Typography>

                    {!isUser && (
                      <Stack direction="row" spacing={1} alignItems="center">
                        {/* Badge RAG si aplica */}
                        {ragSources.length > 0 && (
                          <Tooltip title={`${ragSources.length} fuente(s) local(es) usadas`}>
                            <Chip
                              size="small"
                              icon={<ScienceIcon sx={{ fontSize: 16 }} />}
                              label={`RAG • ${ragSources.length}`}
                              sx={{
                                height: 24,
                                color: "#B3E5FC",
                                borderColor: "rgba(179,229,252,0.35)",
                                background: "rgba(3,169,244,0.12)"
                              }}
                              variant="outlined"
                            />
                          </Tooltip>
                        )}

                        {/* Botón 3 puntos => abre modal con mensajes internos del turno */}
                        {Array.isArray(msg.internal_messages) && msg.internal_messages.length > 0 && (
                          <Tooltip title="Ver mensajes internos de este turno">
                            <IconButton
                              size="small"
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedInternalMessages(msg.internal_messages);
                                setSelectedRagSources(ragSources);
                                setOpenDialog(true);
                              }}
                              sx={{ color: "rgba(255,255,255,0.8)", "&:hover": { color: "#90caf9" } }}
                            >
                              <MoreHorizIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                      </Stack>
                    )}
                  </Box>

                  {/* Respuesta */}
                  <Typography variant="body1" sx={{ color: "white", whiteSpace: "pre-wrap", opacity: msg.pending ? 0.75 : 1 }}>
                    {msg.pending ? " " : (msg.text || "")}
                  </Typography>

                  {/* Loader */}
                  {msg.pending && (
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.5 }}>
                      <CircularProgress size={16} />
                      <Typography variant="caption" sx={{ color: "#CFD8DC" }}>generando respuesta…</Typography>
                    </Box>
                  )}

                  {/* Mermaid */}
                  {msg.mermaidCode && (
                    <Box sx={{ mt: 2 }}>
                      <MermaidChart chart={msg.mermaidCode} />
                    </Box>
                  )}

                  {/* Imágenes del usuario */}
                  {msg.images?.length > 0 && (
                    <Box className="image-container" sx={{ mt: 1 }}>
                      {msg.images.map((src, i) => (
                        <Box key={`${msg.id}-img-${i}`} component="img" src={src} className="message-image" alt={`img-${i}`} />
                      ))}
                    </Box>
                  )}

                  {/* Separador si hay extras */}
                  {!isUser && (roles.length > 0 || (msg.session_id && msg.message_id)) && (
                    <Divider sx={{ my: 1.2, opacity: 0.08 }} />
                  )}

                  {/* Chips de roles (sin duplicados) */}
                  {!isUser && roles.length > 0 && (
                    <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
                      {roles.map(([name, count]) => (
                        <Chip
                          key={`${msg.id}-${name}`}
                          label={`${name}${count > 1 ? ` ×${count}` : ""}`}
                          size="small"
                          variant="outlined"
                          sx={{
                            borderColor: "rgba(255,255,255,0.22)",
                            color: "rgba(255,255,255,0.9)",
                            background: "rgba(255,255,255,0.05)",
                            height: 24
                          }}
                        />
                      ))}
                    </Stack>
                  )}

                  {/* Rating */}
                  {!isUser && msg.session_id && msg.message_id && (
                    <Box sx={{ display: "flex", justifyContent: "flex-end", mt: 1, opacity: ratedMessages.has(ratedKey) ? 0.5 : 1 }}>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.75)", mr: 1, alignSelf: "center" }}>
                        ¿Fue útil esta respuesta?
                      </Typography>
                      <IconButton
                        size="small"
                        onClick={(e) => { e.stopPropagation(); handleRating(msg.session_id, msg.message_id, true); }}
                        disabled={ratedMessages.has(ratedKey)}
                        sx={{ color: "rgba(255,255,255,0.75)", "&:hover": { color: "#4caf50" } }}
                      >
                        <ThumbUpIcon fontSize="small" />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={(e) => { e.stopPropagation(); handleRating(msg.session_id, msg.message_id, false); }}
                        disabled={ratedMessages.has(ratedKey)}
                        sx={{ color: "rgba(255,255,255,0.75)", "&:hover": { color: "#f44336" } }}
                      >
                        <ThumbDownIcon fontSize="small" />
                      </IconButton>
                    </Box>
                  )}
                </Box>
              </ListItem>
            );
          })}
        </List>
      </Paper>

      {/* Chips de imágenes */}
      {attachedImages.length > 0 && (
        <Stack direction="row" spacing={1} sx={{ mt: 2, mb: 2, ml: 1 }}>
          {attachedImages.map((img, index) => (
            <Chip
              key={`chip-${index}-${img.name}`}
              icon={<ImageIcon />}
              label={img.name.length > 18 ? img.name.substring(0, 16) + "…" : img.name}
              onDelete={() => removeImage(index)}
              sx={{ maxWidth: 220, "& .MuiChip-label": { whiteSpace: "nowrap", color: "white" } }}
            />
          ))}
        </Stack>
      )}

      {/* Input */}
      <Box className="input-container">
        <TextField
          className="input-field"
          fullWidth
          variant="outlined"
          placeholder="Escribe un mensaje..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage()}
          InputProps={{
            endAdornment: (
              <IconButton className="attach-button" onClick={() => fileInputRef.current?.click()} disabled={attachedImages.length >= 2}>
                <Badge badgeContent={attachedImages.length} color="primary">
                  <AttachFileIcon />
                </Badge>
              </IconButton>
            )
          }}
        />
        <input type="file" multiple accept="image/*" ref={fileInputRef} onChange={handleImageUpload} style={{ display: "none" }} />
        <Button className="send-button" variant="contained" onClick={sendMessage}>ENVIAR</Button>
      </Box>

      {/* Modal: mensajes internos + fuentes RAG */}
      <Dialog
        className="nodes-dialog"
        open={openDialog}
        onClose={() => setOpenDialog(false)}
        fullWidth
        maxWidth="md"
        PaperProps={{
          sx: {
            background: "rgba(17,17,17,0.96)",
            border: "1px solid rgba(255,255,255,0.08)",
            boxShadow: "0 20px 60px rgba(0,0,0,0.5)"
          }
        }}
      >
        <DialogTitle sx={{ pr: 6 }}>
          Mensajes Internos (turno actual)
          <IconButton aria-label="close" onClick={() => setOpenDialog(false)} sx={{ position: "absolute", right: 8, top: 8 }}>
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent dividers sx={{ borderColor: "rgba(255,255,255,0.08)" }}>
          {/* Fuentes RAG si hubo */}
          {selectedRagSources.length > 0 && (
            <Box sx={{ mb: 2, p: 1.2, border: "1px dashed rgba(179,229,252,0.35)", borderRadius: 1.5, background: "rgba(3,169,244,0.06)" }}>
              <Typography variant="subtitle2" sx={{ color: "#B3E5FC", display: "flex", alignItems: "center", gap: 1 }}>
                <ScienceIcon fontSize="small" /> Fuentes locales usadas
              </Typography>
              <List dense sx={{ mt: 0.5 }}>
                {selectedRagSources.map((s, i) => (
                  <ListItem key={`src-${i}`} sx={{ py: 0 }}>
                    <ListItemText
                      primary={`• ${s}`}
                      primaryTypographyProps={{ color: "white", fontSize: 13 }}
                    />
                  </ListItem>
                ))}
              </List>
            </Box>
          )}

          {/* Conversación interna */}
          <List dense>
            {selectedInternalMessages.map((m, i) => (
              <ListItem key={`internal-${i}`} sx={{ alignItems: "flex-start" }}>
                <ListItemText
                  primary={`${m.name || m.role || "system"}:`}
                  secondary={m.content}
                  primaryTypographyProps={{ color: "#90caf9", fontWeight: 600 }}
                  secondaryTypographyProps={{ color: "white", whiteSpace: "pre-wrap" }}
                />
              </ListItem>
            ))}
          </List>
        </DialogContent>
      </Dialog>
    </Box>
  );
}
