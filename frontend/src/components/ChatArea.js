import React, { useRef, useEffect } from 'react';
import { Bot, User, Info, AlertCircle, Loader2 } from 'lucide-react';
import './ChatArea.css';
import Stage1Card from './Stage1Card';
import Stage2Card from './Stage2Card';

export default function ChatArea({ messages, loading }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  return (
    <div className="chat-area">
      {messages.length === 0 && (
        <div className="chat-empty">
          <Bot size={48} strokeWidth={1.2} />
          <h3>LexIR Legal Assistant</h3>
          <p>Submit a case above to begin the 3-stage analysis pipeline.</p>
        </div>
      )}

      {messages.map((msg) => {
        // Stage result cards
        if (msg.role === 'stage') {
          if (msg.meta.stage === 1) {
            return <Stage1Card key={msg.id} data={msg.meta.data} />;
          }
          if (msg.meta.stage === 2) {
            return <Stage2Card key={msg.id} data={msg.meta.data} />;
          }
          return null;
        }

        // System status messages
        if (msg.role === 'system') {
          return (
            <div key={msg.id} className={`chat-bubble system ${msg.meta.isError ? 'system-error' : ''}`}>
              {msg.meta.isError
                ? <AlertCircle size={14} className="bubble-icon" />
                : <Info size={14} className="bubble-icon" />
              }
              <span>{msg.content}</span>
            </div>
          );
        }

        // User messages
        if (msg.role === 'user') {
          return (
            <div key={msg.id} className="chat-bubble user">
              <div className="bubble-avatar user-avatar"><User size={16} /></div>
              <div className="bubble-content">
                <p>{msg.content}</p>
              </div>
            </div>
          );
        }

        // Assistant (Q&A) messages
        if (msg.role === 'assistant') {
          return (
            <div key={msg.id} className="chat-bubble assistant">
              <div className="bubble-avatar bot-avatar"><Bot size={16} /></div>
              <div className="bubble-content">
                {msg.content.split('\n').map((line, i) => (
                  <p key={i}>{line || '\u00A0'}</p>
                ))}
                {msg.meta.precedents_used > 0 && (
                  <span className="precedent-count">
                    Based on {msg.meta.precedents_used} precedent(s)
                  </span>
                )}
              </div>
            </div>
          );
        }

        return null;
      })}

      {/* Typing indicator */}
      {loading && (
        <div className="chat-bubble system loading-bubble">
          <Loader2 size={16} className="spin" />
          <span>Analyzing...</span>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
