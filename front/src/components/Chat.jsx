import React, { useEffect, useMemo, useRef, useState } from "react";
import MermaidChart from "./MermaidChart";
import {
  Box, Paper, List, ListItem, TextField, Button, Typography, Dialog,
  DialogTitle, DialogContent, ListItemText, IconButton, Badge, Chip, Stack,
  Tooltip, Divider, CircularProgress, Drawer, ListItemButton
} from "@mui/material";
import AttachFileIcon from "@mui/icons-material/AttachFile";
import CloseIcon from "@mui/icons-material/Close";
import ImageIcon from "@mui/icons-material/Image";
import ThumbUpIcon from "@mui/icons-material/ThumbUp";
import ThumbDownIcon from "@mui/icons-material/ThumbDown";
import MoreHorizIcon from "@mui/icons-material/MoreHoriz";
import EditOutlinedIcon from "@mui/icons-material/EditOutlined";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline";
import ScienceIcon from "@mui/icons-material/Science";
import "../styles/chat.css";

/* ======================= Utils ======================= */
const uuid = () =>
  (crypto?.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random()));

const STORAGE_KEYS = {
  SESSIONS: "arquia.sessions",
  MESSAGES: (sid) => `arquia.chat.${sid}`,
};

const DISCLAIMER =
  "ArquIA es un asistente para arquitectas y arquitectos de software: acelera análisis, tácticas y diagramas. Puede cometer errores; verifica siempre la información importante.";

const titleFrom = (text) => {
  const t = (text || "").trim();
  if (!t) return "New chat";
  const firstLine = t.split("\n")[0].slice(0, 60);
  return firstLine || "New chat";
};

// storage helpers
const loadSessions = () => {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEYS.SESSIONS) || "[]"); } catch { return []; }
};
const saveSessions = (arr) => localStorage.setItem(STORAGE_KEYS.SESSIONS, JSON.stringify(arr));
const loadChat = (sid) => {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEYS.MESSAGES(sid)) || "[]"); } catch { return []; }
};
const saveChat = (sid, msgs) => localStorage.setItem(STORAGE_KEYS.MESSAGES(sid), JSON.stringify(msgs));

/* Parse “Next:” fallback si el backend no envía suggestions[] */
const parseNextFromText = (text = "") => {
  const m = text.match(/^\s*Next:\s*([\s\S]*)$/im);
  if (!m) return [];
  const lines = m[1]
    .split("\n")
    .map((ln) => ln.replace(/^[\s\-•]+/, "").trim())
    .filter(Boolean);
  // corta en el primer bloque vacío
  const cleaned = [];
  for (const ln of lines) {
    if (!ln) break;
    cleaned.push(ln);
    if (cleaned.length >= 8) break;
  }
  return cleaned;
};

/* RAG helpers para el modal */
const summarizeRoles = (internal = []) => {
  const counts = new Map();
  const order = ["supervisor", "researcher", "evaluator", "creator", "asr_recommender", "unifier"];
  internal.forEach((m) => {
    const k = (m.name || m.role || "other").toLowerCase();
    counts.set(k, (counts.get(k) || 0) + 1);
  });
  const ordered = order.filter(k => counts.has(k)).map(k => [k, counts.get(k)]);
  const rest = [...counts.entries()].filter(([k]) => !order.includes(k));
  return [...ordered, ...rest];
};
const extractRagSources = (internal = []) => {
  const out = [];
  for (const m of internal) {
    const text = String(m.content || "");
    const idx = text.indexOf("\nSOURCES:\n");
    if (idx !== -1) {
      const lines = text.substring(idx + "\nSOURCES:\n".length).split("\n");
      for (const ln of lines) { const t = ln.trim(); if (t.startsWith("- ")) out.push(t.slice(2)); }
    }
  }
  return out;
};

/* ======================= Chat ======================= */
export default function Chat() {
  // Drawer (escucha al header global)
  const [drawerOpen, setDrawerOpen] = useState(false);
  useEffect(() => {
    const handler = () => setDrawerOpen((p) => !p);
    window.addEventListener("arquia-toggle-drawer", handler);
    return () => window.removeEventListener("arquia-toggle-drawer", handler);
  }, []);

  // Sessions
  const [sessions, setSessions] = useState(loadSessions());
  const [sessionId, setSessionId] = useState(() => sessions?.[0]?.id || uuid());

  // Chat state
  const [messages, setMessages] = useState(() => loadChat(sessionId));
  const [input, setInput] = useState("");
  const [attachedImages, setAttachedImages] = useState([]);
  const [openDialog, setOpenDialog] = useState(false);
  const [selectedInternalMessages, setSelectedInternalMessages] = useState([]);
  const [selectedRagSources, setSelectedRagSources] = useState([]);
  const [ratedMessages, setRatedMessages] = useState(new Set());
  const fileInputRef = useRef(null);
  const requestSeq = useRef(0);

  // ¿hay respuesta pendiente? -> bloquea UI
  const isBusy = useMemo(() => messages.some((m) => m.pending), [messages]);

  useEffect(() => {
    if (sessions.length === 0) {
      const first = { id: sessionId, title: "New chat", createdAt: Date.now() };
      const arr = [first];
      setSessions(arr); saveSessions(arr); saveChat(sessionId, []);
    } else {
      setMessages(loadChat(sessionId));
    }
  }, []);

  useEffect(() => { setMessages(loadChat(sessionId)); }, [sessionId]);

  // sessions ops
  const createSession = () => {
    if (isBusy) return;
    const id = uuid();
    const s = { id, title: "New chat", createdAt: Date.now() };
    const arr = [s, ...sessions];
    setSessions(arr); saveSessions(arr); setSessionId(id); saveChat(id, []); setDrawerOpen(false);
  };
  const renameSession = (id, title) => {
    const arr = sessions.map((s) => (s.id === id ? { ...s, title } : s));
    setSessions(arr); saveSessions(arr);
  };
  const deleteSession = (id) => {
    if (isBusy) return;
    const arr = sessions.filter((s) => s.id !== id);
    setSessions(arr); saveSessions(arr);
    localStorage.removeItem(STORAGE_KEYS.MESSAGES(id));
    if (id === sessionId) {
      const next = arr[0]?.id || uuid();
      if (!arr[0]) {
        const ns = { id: next, title: "New chat", createdAt: Date.now() };
        const arr2 = [ns];
        setSessions(arr2); saveSessions(arr2); saveChat(next, []);
      }
      setSessionId(next);
    }
  };

  // ------- envío ----------
  const sendMessage = async (overrideText) => {
    if (isBusy) return; // bloquear envío cuando está “pensando”

    const textToSend = (overrideText ?? input).trim();
    if (!textToSend && attachedImages.length === 0) return;

    if (messages.length === 0) renameSession(sessionId, titleFrom(textToSend));

    const userMsg = {
      id: uuid(),
      sender: "usuario",
      text: textToSend,
      images: attachedImages.map((img) => img.preview),
    };
    const pendingId = "pending-" + uuid();
    const pending = { id: pendingId, sender: "respuesta", text: "", pending: true };

    const optimistic = [...messages, userMsg, pending];
    setMessages(optimistic);
    saveChat(sessionId, optimistic);

    const form = new FormData();
    form.append("message", textToSend);
    form.append("session_id", sessionId);
    attachedImages.forEach((img, index) => form.append(`image${index + 1}`, img.file));

    // limpiar input/adjuntos
    setInput("");
    setAttachedImages([]);
    if (fileInputRef.current) fileInputRef.current.value = "";

    const seq = ++requestSeq.current;
    try {
      const resp = await fetch("http://localhost:8000/message", { method: "POST", body: form });
      const data = await resp.json();
      if (seq !== requestSeq.current) return;

      // usamos texto tal cual del backend (Answer/References/Next)
      const textOut = data?.endMessage || "—";

      const rendered = optimistic.map((m) =>
        m.id === pendingId
          ? {
              ...m,
              pending: false,
              text: textOut,
              internal_messages: Array.isArray(data?.messages) ? data.messages : [],
              mermaidCode: data?.mermaidCode || "",
              session_id: data?.session_id || sessionId,
              message_id: data?.message_id,
              suggestions: Array.isArray(data?.suggestions) && data.suggestions.length > 0
                ? data.suggestions
                : parseNextFromText(textOut) // fallback si no vino el array
            }
          : m
      );
      setMessages(rendered);
      saveChat(sessionId, rendered);
    } catch (e) {
      const rendered = optimistic.map((m) =>
        m.id === pendingId ? { ...m, pending: false, text: "⚠️ Error generating the answer." } : m
      );
      setMessages(rendered);
      saveChat(sessionId, rendered);
      console.error(e);
    }
  };

  // chips -> manda la sugerencia como prompt
  const onSuggestionClick = (sugg) => {
    if (isBusy) return;
    sendMessage(sugg);
  };

  // feedback
  const handleThumbClick = (sid, mid, thumbs_up, thumbs_down) => {
    const form = new FormData();
    form.append("session_id", sid);
    form.append("message_id", mid);
    form.append("thumbs_up", thumbs_up);
    form.append("thumbs_down", thumbs_down);
    fetch("http://localhost:8000/feedback", { method: "POST", body: form });
  };
  const handleRating = (sid, mid, isUp) => {
    const key = `${sid}-${mid}`;
    if (ratedMessages.has(key)) return;
    const next = new Set(ratedMessages); next.add(key); setRatedMessages(next);
    handleThumbClick(sid, mid, isUp ? 1 : 0, isUp ? 0 : 1);
  };

  // images
  const handleImageUpload = (e) => {
    if (isBusy) return;
    const files = Array.from(e.target.files);
    if (files.length + attachedImages.length > 2) { alert("Solo puedes adjuntar hasta 2 imágenes."); return; }
    const picks = files.map((f) => ({ file: f, preview: URL.createObjectURL(f), name: f.name }));
    setAttachedImages((p) => [...p, ...picks]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };
  const removeImage = (i) => { if (!isBusy) setAttachedImages((p) => p.filter((_, idx) => idx !== i)); };

  /* ======================= UI ======================= */
  const bubbleSx = (isUser) => ({
    width: "100%",
    borderRadius: 12,
    padding: "14px 16px",
    background: isUser ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.10)",
    boxShadow: "0 8px 22px rgba(0,0,0,0.28)",
    position: "relative",
    color: "#fff",
  });
  const headerSx = { display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1 };
  const headerTitle = (msg) => (msg.sender === "usuario" ? "Tú" : msg.pending ? "Asistente (pensando…)" : "Asistente");

  // Drawer UI
  const DrawerContent = (
    <Box sx={{ width: 300, height: "100%", background: "#151515", color: "#fff", display: "flex", flexDirection: "column" }}>
      <Box sx={{ px: 2, py: 1.5, borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
        <Typography variant="subtitle1" sx={{ color: "#fff", fontWeight: 700 }}>Conversations</Typography>
      </Box>

      <Box sx={{ flex: 1, overflowY: "auto", py: 1 }}>
        {sessions.map((s) => (
          <ListItem key={s.id} disableGutters sx={{
            px: 1,
            "&:hover": { background: "rgba(255,255,255,0.06)" },
            background: s.id === sessionId ? "rgba(3,169,244,0.14)" : "transparent",
            borderLeft: s.id === sessionId ? "3px solid #03A9F4" : "3px solid transparent",
          }}>
            <ListItemButton onClick={() => { if (!isBusy) { setSessionId(s.id); setDrawerOpen(false); } }}>
              <ChatBubbleOutlineIcon sx={{ mr: 1.2, color: "#B3E5FC" }} />
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Typography noWrap sx={{ color: "#fff", fontSize: 14 }}>{s.title || "New chat"}</Typography>
                <Typography noWrap sx={{ color: "rgba(255,255,255,0.55)", fontSize: 11 }}>
                  {new Date(s.createdAt).toLocaleString()}
                </Typography>
              </Box>
              <Tooltip title="Rename">
                <IconButton size="small" onClick={(e) => {
                  e.stopPropagation();
                  const t = prompt("Rename conversation", s.title || "New chat");
                  if (t !== null) renameSession(s.id, t.trim() || "New chat");
                }} sx={{ color: "rgba(255,255,255,0.8)" }}>
                  <EditOutlinedIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              <Tooltip title="Delete">
                <IconButton size="small" onClick={(e) => { e.stopPropagation(); if (confirm("Delete this conversation?")) deleteSession(s.id); }} sx={{ color: "rgba(255,255,255,0.8)" }}>
                  <DeleteOutlineIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </ListItemButton>
          </ListItem>
        ))}
      </Box>

      <Box sx={{ p: 1.2, borderTop: "1px solid rgba(255,255,255,0.08)" }}>
        <Button
          fullWidth
          onClick={createSession}
          disabled={isBusy}
          variant="outlined"
          sx={{ color: "#fff", borderColor: "rgba(255,255,255,0.28)", "&:hover": { borderColor: "#90CAF9", background: "rgba(144,202,249,0.10)" } }}
        >
          NEW CHAT
        </Button>
      </Box>
    </Box>
  );

  return (
    <Box className="chat-root" sx={{ background: "#111", minHeight: "100vh" }}>
      <Drawer anchor="left" open={drawerOpen} onClose={() => setDrawerOpen(false)} PaperProps={{ sx: { background: "transparent" } }}>
        {DrawerContent}
      </Drawer>

      <Paper className="messages-box" sx={{ background: "transparent" }}>
        <Typography variant="caption" sx={{ color: "#cfcfcf", px: 2, pt: 1, display: "block", opacity: 0.65 }}>
          session: {sessionId}
        </Typography>

        <List sx={{ px: 2, py: 1 }}>
          {messages.map((msg) => {
            const isUser = msg.sender === "usuario";
            const roles = !isUser ? summarizeRoles(msg.internal_messages) : [];
            const ragSources = !isUser ? extractRagSources(msg.internal_messages) : [];
            const ratedKey = `${msg.session_id}-${msg.message_id}`;

            return (
              <ListItem key={msg.id} disableGutters sx={{ mb: 2 }}>
                <Box sx={bubbleSx(isUser)}>
                  <Box sx={headerSx}>
                    <Typography variant="subtitle2" sx={{ color: "#E8EAF6", fontWeight: 600 }}>
                      {headerTitle(msg)}
                    </Typography>
                    {!isUser && (
                      <Stack direction="row" spacing={1} alignItems="center">
                        {ragSources.length > 0 && (
                          <Tooltip title={`${ragSources.length} local source(s)`}>
                            <Chip
                              size="small"
                              icon={<ScienceIcon sx={{ fontSize: 16 }} />}
                              label={`RAG • ${ragSources.length}`}
                              sx={{ height: 24, color: "#B3E5FC", borderColor: "rgba(179,229,252,0.35)", background: "rgba(3,169,244,0.12)" }}
                              variant="outlined"
                            />
                          </Tooltip>
                        )}
                        {Array.isArray(msg.internal_messages) && msg.internal_messages.length > 0 && (
                          <Tooltip title="View internal messages">
                            <IconButton
                              size="small"
                              onClick={(e) => { e.stopPropagation(); setSelectedInternalMessages(msg.internal_messages || []); setSelectedRagSources(ragSources); setOpenDialog(true); }}
                              sx={{ color: "rgba(255,255,255,0.8)", "&:hover": { color: "#90caf9" } }}
                            >
                              <MoreHorizIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                      </Stack>
                    )}
                  </Box>

                  <Typography variant="body1" sx={{ color: "white", whiteSpace: "pre-wrap", opacity: msg.pending ? 0.75 : 1 }}>
                    {msg.pending ? " " : (msg.text || "")}
                  </Typography>

                  {msg.pending && (
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.5 }}>
                      <CircularProgress size={16} />
                      <Typography variant="caption" sx={{ color: "##CFD8DC" }}>generating response…</Typography>
                    </Box>
                  )}

                  {msg.mermaidCode && (
                    <Box sx={{ mt: 2 }}>
                      <MermaidChart chart={msg.mermaidCode} />
                    </Box>
                  )}

                  {msg.images?.length > 0 && (
                    <Box className="image-container" sx={{ mt: 1 }}>
                      {msg.images.map((src, i) => (
                        <Box key={`${msg.id}-img-${i}`} component="img" src={src} className="message-image" alt={`img-${i}`} />
                      ))}
                    </Box>
                  )}

                  {/* Chips “Next” */}
                  {!isUser && Array.isArray(msg.suggestions) && msg.suggestions.length > 0 && (
                    <>
                      <Divider sx={{ my: 1.2, opacity: 0.08 }} />
                      <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
                        {msg.suggestions.map((s, i) => (
                          <Chip
                            key={`${msg.id}-sugg-${i}`}
                            label={s}
                            onClick={() => onSuggestionClick(s)}
                            disabled={isBusy}
                            size="small"
                            sx={{
                              borderColor: "rgba(3,169,244,0.35)",
                              color: "#B3E5FC",
                              background: "rgba(3,169,244,0.10)",
                              height: 26
                            }}
                            variant="outlined"
                          />
                        ))}
                      </Stack>
                    </>
                  )}

                  {/* Roles y rating */}
                  {!isUser && (roles.length > 0 || (msg.session_id && msg.message_id)) && (
                    <Divider sx={{ my: 1.2, opacity: 0.08 }} />
                  )}

                  {!isUser && roles.length > 0 && (
                    <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
                      {roles.map(([name, count]) => (
                        <Chip
                          key={`${msg.id}-${name}`}
                          label={`${name}${count > 1 ? ` ×${count}` : ""}`}
                          size="small"
                          variant="outlined"
                          sx={{ borderColor: "rgba(255,255,255,0.22)", color: "rgba(255,255,255,0.9)", background: "rgba(255,255,255,0.05)", height: 24 }}
                        />
                      ))}
                    </Stack>
                  )}

                  {!isUser && msg.session_id && msg.message_id && (
                    <Box sx={{ display: "flex", justifyContent: "flex-end", mt: 1, opacity: ratedMessages.has(ratedKey) ? 0.5 : 1 }}>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.75)", mr: 1, alignSelf: "center" }}>
                        Was this helpful?
                      </Typography>
                      <IconButton size="small" onClick={(e) => { e.stopPropagation(); handleRating(msg.session_id, msg.message_id, true); }}
                        disabled={ratedMessages.has(ratedKey)} sx={{ color: "rgba(255,255,255,0.75)", "&:hover": { color: "#4caf50" } }}>
                        <ThumbUpIcon fontSize="small" />
                      </IconButton>
                      <IconButton size="small" onClick={(e) => { e.stopPropagation(); handleRating(msg.session_id, msg.message_id, false); }}
                        disabled={ratedMessages.has(ratedKey)} sx={{ color: "rgba(255,255,255,0.75)", "&:hover": { color: "#f44336" } }}>
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

      {/* Input + acciones */}
      <Box className="input-container">
        <TextField
          className="input-field"
          fullWidth
          variant="outlined"
          placeholder={isBusy ? "Esperando respuesta…" : "Escribe un mensaje..."}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (!isBusy && e.key === "Enter" && !e.shiftKey) sendMessage(); }}
          disabled={isBusy}
          InputProps={{
            endAdornment: (
              <IconButton onClick={() => !isBusy && fileInputRef.current?.click()} disabled={attachedImages.length >= 2 || isBusy} sx={{ color: "#fff" }}>
                <Badge badgeContent={attachedImages.length} color="primary">
                  <AttachFileIcon />
                </Badge>
              </IconButton>
            ),
          }}
          sx={{ "& .MuiInputBase-root": { background: "#1d1d1d", color: "#fff", borderRadius: 1 } }}
        />
        <input type="file" multiple accept="image/*" ref={fileInputRef} onChange={handleImageUpload} style={{ display: "none" }} />
        <Button className="send-button" variant="contained" onClick={() => sendMessage()} disabled={isBusy}>
          {isBusy ? "ENVIANDO..." : "ENVIAR"}
        </Button>
      </Box>

      {/* Modal: mensajes internos */}
      <Dialog
        className="nodes-dialog"
        open={openDialog}
        onClose={() => setOpenDialog(false)}
        fullWidth
        maxWidth="md"
        PaperProps={{ sx: { background: "rgba(17,17,17,0.96)", border: "1px solid rgba(255,255,255,0.08)", boxShadow: "0 20px 60px rgba(0,0,0,0.5)" } }}
      >
        <DialogTitle sx={{ pr: 6, color: "#fff" }}>
          Mensajes Internos (turno actual)
          <IconButton aria-label="close" onClick={() => setOpenDialog(false)} sx={{ position: "absolute", right: 8, top: 8, color: "#fff" }}>
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent dividers sx={{ borderColor: "rgba(255,255,255,0.08)" }}>
          {selectedRagSources.length > 0 && (
            <Box sx={{ mb: 2, p: 1.2, border: "1px dashed rgba(179,229,252,0.35)", borderRadius: 1.5, background: "rgba(3,169,244,0.06)" }}>
              <Typography variant="subtitle2" sx={{ color: "#B3E5FC", display: "flex", alignItems: "center", gap: 1 }}>
                <ScienceIcon fontSize="small" /> Local sources
              </Typography>
              <List dense sx={{ mt: 0.5 }}>
                {selectedRagSources.map((s, i) => (
                  <ListItem key={`src-${i}`} sx={{ py: 0 }}>
                    <ListItemText primary={`• ${s}`} primaryTypographyProps={{ color: "white", fontSize: 13 }} />
                  </ListItem>
                ))}
              </List>
            </Box>
          )}
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
            <Box
        component="footer"
        sx={{
          px: 2,
          py: 1.5,
          textAlign: "center",
          color: "rgba(255,255,255,0.65)",
          fontSize: 12,
          borderTop: "1px dashed rgba(255,255,255,0.12)",
          maxWidth: 980,
          mx: "auto",
          mt: 1.5,
        }}
      >
        {DISCLAIMER}
      </Box>

    </Box>
  );
}
