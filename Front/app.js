// REPLACE with your actual backend URL if different
const BACKEND_URL = "http://127.0.0.1:8000/chat"; 

const chat = document.getElementById("chat");
const input = document.getElementById("input");
const send = document.getElementById("send");

let currentSessionId = null;

function add(role, text) {
  const msg = document.createElement("div");
  msg.className = "msg " + role;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  msg.appendChild(bubble);
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

async function sendMessage() {
  const text = input.value.trim();
  if (text === "") return;

  // --- 1. DISABLE CONTROLS ---
  input.disabled = true;
  send.disabled = true;
  send.textContent = "..."; // Optional: Change text to show loading

  add("user", text);
  input.value = "";

  const payload = { message: text };
  
  if (currentSessionId) {
    payload.session_id = currentSessionId;
  }

  try {
    const res = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();
    
    if (data.session_id) {
        currentSessionId = data.session_id;
    }

    add("bot", data.response); 

  } catch (error) {
    console.error(error);
    add("bot", "Error connecting to server.");
  } finally {
    // --- 2. RE-ENABLE CONTROLS (Always runs, success or error) ---
    input.disabled = false;
    send.disabled = false;
    send.textContent = "SEND"; 
    input.focus(); // Put cursor back in input for next message
  }
}

send.onclick = sendMessage;

input.addEventListener("keydown", function (e) {
  if (e.key === "Enter") sendMessage();
});