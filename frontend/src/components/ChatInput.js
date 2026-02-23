import React, { useState } from 'react';
import { Send } from 'lucide-react';

export default function ChatInput({ onSend, disabled, placeholder }) {
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!text.trim() || disabled) return;
    onSend(text.trim());
    setText('');
  };

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <input
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={placeholder || 'Ask a legal question about the case...'}
        disabled={disabled}
      />
      <button type="submit" disabled={disabled || !text.trim()} className="btn btn-send">
        <Send size={18} />
      </button>
    </form>
  );
}
