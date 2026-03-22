# 🎨 Frontend Design Concept: RAG-Enhanced PDF Chat

This document outlines the visual and interactive design for the new **React + Vite** frontend. The goal is a premium, professional, and highly responsive interface that feels like a true AI research assistant.

---

## 🌓 Core Theme: "Indigo Night"
- **Primary Colors**: Deep Slate (`#0f172a`), Indigo (`#6366f1`), and Zinc (`#18181b`).
- **Surface**: Glassmorphism with subtle blurs (`backdrop-blur-md`).
- **Typography**: Inter or Outfit (Modern Sans-Serif).

---

## 🛠️ Layout Structure

### 1. **Sidebar (Navigation & Document Management)**
- **Glass-style Sidebar**: A fixed left-hand panel for control.
- **Logo Area**: Animated logo with a subtle pulse when the AI is "thinking".
- **Upload Dropzone**: A dotted-border area with "Drag & Drop PDF".
- **Category Badge**: A glowing indicator of the auto-detected category (e.g., Blue for Research Paper, Amber for Receipt).
- **Session Controls**: A simple "Clear Session" button with a red hover effect.

### 2. **Main Chat Area**
- **Centered Vessel**: The chat follows a clean, centered column for readability.
- **Message Bubbles**:
    - **User**: Minimalist, right-aligned, slate background.
    - **Assistant**: Left-aligned, subtle gradient border, markdown supported.
- **Source Citations**: Small, clickable "cards" or "pills" at the bottom of AI responses. Hovering reveals a preview of the source text.
- **Loading State**: Shimmering skeleton screens while the AI processes.

---

## ✨ Interactive Features

### 1. **Smooth Micro-Animations**
- **Framer Motion**: Messages should slide up gently when they appear.
- **Progress Bar**: A slim indigo bar at the top of the chat area during document processing.
- **Hover Effects**: Buttons and cards should have a scale-up effect (+2%) when hovered.

### 2. **Rich Markdown Support**
- **Syntax Highlighting**: Using `react-syntax-highlighter` for any code snippets found in PDFs.
- **Table Formatting**: Clean, bordered tables for structured data (like receipts).
- **LaTeX Support**: If the PDF contains math, use `react-katex` for rendering formulas.

### 3. **Responsive Design**
- **Mobile First**: Sidebar collapses into a "hamburger" menu on smaller screens.
- **Auto-scroll**: The chat automatically follows new messages with a smooth "spring" animation.

---

## 📱 Component Breakdown (React)
- `Sidebar.tsx`: Global navigation and file management.
- `ChatContainer.tsx`: Orchestrates the message list and input.
- `MessageBubble.tsx`: Individual user/assistant messages.
- `UploadZone.tsx`: The interactive PDF drop area.
- `SourceCard.tsx`: Expandable preview of retrieved document chunks.
- `Header.tsx`: Contextual information (App name, current document).
