const chatBox = document.getElementById("chat-box");
const chatHistory = document.getElementById("chat-history");
const newChatBtn = document.getElementById("new-chat-btn");
const sendBtn = document.getElementById("send-btn");
const userInput = document.getElementById("user-msg");
const micBtn = document.getElementById("mic-btn");
const pauseBtn = document.getElementById("pause-btn");
const cancelBtn = document.getElementById("cancel-btn");


let chatConversations = []; // All previous conversations
let currentConversation = { id: Date.now(), messages: [] }; // Current conversation


cancelBtn.addEventListener("click", () => {
    if (speechSynthesis.speaking) {
        speechSynthesis.cancel(); // stops all ongoing speech
    }
});

pauseBtn.addEventListener("click", () => {
    if (speechSynthesis.speaking && !speechSynthesis.paused) {
        speechSynthesis.pause();
        pauseBtn.textContent = "▶"; // change icon to resume
    } else if (speechSynthesis.paused) {
        speechSynthesis.resume();
        pauseBtn.textContent = "⏸";
    }
});



// --- Speech Recognition ---
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;

    micBtn.addEventListener("click", () => recognition.start());

    recognition.addEventListener("result", (event) => {
        const transcript = event.results[0][0].transcript;
        userInput.value = transcript;
        sendMessage();
    });

    recognition.addEventListener("end", () => recognition.stop());
} else {
    console.warn("Speech Recognition not supported in this browser.");
}

// --- Append message to chat box ---
function appendMessage(sender, message) {
    const wrapper = document.createElement("div");
    wrapper.classList.add("message-wrapper");

    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message-card", sender);

    // If system message is Urdu, add lang attribute for RTL
    if (sender === "system" && isUrdu(message)) {
        msgDiv.lang = "ur";        // used in HTML for CSS
        msgDiv.style.direction = "rtl";
        msgDiv.style.textAlign = "right";
    } else if (sender === "system") {
        msgDiv.lang = "en";
        msgDiv.style.direction = "ltr";
        msgDiv.style.textAlign = "left";
    }

    msgDiv.innerHTML = message;
    wrapper.appendChild(msgDiv);

    if (sender === "user") {
        const actionsDiv = document.createElement("div");
        actionsDiv.classList.add("msg-actions-wrapper");

        const copyIcon = document.createElement("span");
        copyIcon.innerHTML = "📋";
        copyIcon.title = "Copy message";
        copyIcon.addEventListener("click", () => navigator.clipboard.writeText(msgDiv.innerText));

        const editIcon = document.createElement("span");
        editIcon.innerHTML = "✏️";
        editIcon.title = "Edit message";
        editIcon.addEventListener("click", () => editMessage(wrapper, msgDiv));

        actionsDiv.appendChild(copyIcon);
        actionsDiv.appendChild(editIcon);
        wrapper.appendChild(actionsDiv);
    }

    chatBox.appendChild(wrapper);
    chatBox.scrollTop = chatBox.scrollHeight;
}


// --- Edit message ---
function editMessage(wrapper, msgDiv) {
    const nextSibling = wrapper.nextElementSibling;
    let oldResponse = null;

    if (nextSibling && nextSibling.querySelector(".message-card.system")) {
        oldResponse = nextSibling.querySelector(".message-card.system").innerHTML;
        nextSibling.remove();
    }

    userInput.value = msgDiv.innerText;
    wrapper.remove();
    userInput.focus();
    userInput.dataset.oldResponse = oldResponse || "";
}

// --- Send message ---
function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    appendMessage("user", message);
    currentConversation.messages.push({ sender: "user", message });

    fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
    })
    .then(res => res.json())
    .then(data => {
        appendMessage("system", data.system_msg);
        currentConversation.messages.push({ sender: "system", message: data.system_msg });

        if (data.raw_text) {
            const utterance = new SpeechSynthesisUtterance(data.raw_text);
            utterance.lang = 'en-US';   // English voice
            speechSynthesis.speak(utterance);
        }
    })
    .catch(err => {
        console.error(err);
        appendMessage("system", "Error: Could not get response.");
    });

    userInput.value = "";
}

// --- Add conversation to history ---
function addConversationToHistory(conversation) {
    const li = document.createElement("li");
    li.classList.add("history-card");
    li.innerText = conversation.messages.find(m => m.sender === "user")?.message || "New Chat";
    li.dataset.convId = conversation.id;

    li.addEventListener("click", () => {
        const convId = parseInt(li.dataset.convId);
        const conv = chatConversations.find(c => c.id === convId);
        if (!conv) return;

        chatBox.innerHTML = "";
        conv.messages.forEach(m => appendMessage(m.sender, m.message));
        currentConversation = conv;
    });

    chatHistory.prepend(li);
}

// --- New Chat ---
newChatBtn.addEventListener("click", () => {
    if (currentConversation && currentConversation.messages.length > 0) {
        if (!chatConversations.includes(currentConversation)) {
            chatConversations.push(currentConversation);
            addConversationToHistory(currentConversation);
        }
    }

    chatBox.innerHTML = "";
    userInput.value = "";
    currentConversation = { id: Date.now(), messages: [] };
});

// --- Event listeners ---
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", e => { if (e.key === "Enter") sendMessage(); });
