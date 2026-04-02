import React, { useRef, useEffect, useState } from 'react';
import { Bot, User, Info, AlertCircle, Loader2, Lightbulb, Edit2, Check, X } from 'lucide-react';
import './ChatArea.css';
import Stage1Card from './Stage1Card';
import Stage2Card from './Stage2Card';

function PipelineProgress({ progress }) {
  if (!progress) return null;
  const { s1, s2, s3 } = progress;

  if (s3 === 'ready') return null;

  let label = '';
  if (s1 === 'running') label = 'FIR Analysis — Section mapping...';
  else if (s2 === 'running') label = 'Case Similarity — Fetching precedents...';

  if (!label) return null;

  return (
    <div className="chat-bubble system loading-bubble" aria-live="polite">
      <div className="loading-row">
        <Loader2 size={16} className="spin" />
        <span>{label}</span>
      </div>
    </div>
  );
}
export default function ChatArea({ 
  messages, 
  loading, 
  loadedSession, 
  pipelineProgress, 
  chatTitle, 
  renameSession,
  sessionId 
}) {
  const bottomRef = useRef(null);
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [editedTitle, setEditedTitle] = useState(chatTitle);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading, loadedSession, pipelineProgress]);

  useEffect(() => {
    setEditedTitle(chatTitle);
  }, [chatTitle]);

  const handleSaveTitle = () => {
    if (editedTitle.trim() && editedTitle !== chatTitle) {
      renameSession(sessionId || loadedSession?.sessionId, editedTitle);
    }
    setIsEditingTitle(false);
  };

  const showLiveChat = !loadedSession;
  const qaPairs = loadedSession?.messages || [];

  function renderMessage(msg) {
    if (msg.role === 'stage') {
      if (msg.meta.stage === 1) {
        return <Stage1Card key={msg.id} data={msg.meta.data} />;
      }
      if (msg.meta.stage === 2) {
        return <Stage2Card key={msg.id} data={msg.meta.data} />;
      }
      return null;
    }

    if (msg.role === 'thought') {
      return (
        <div key={msg.id} className="chat-bubble thought">
          <Lightbulb size={12} className="thought-icon" />
          <span>{msg.content}</span>
        </div>
      );
    }

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

    if (msg.role === 'assistant') {
      return (
        <div key={msg.id} className="chat-bubble assistant">
          <div className="bubble-avatar bot-avatar"><Bot size={16} /></div>
          <div className="bubble-content">
            {msg.content.split('\n').map((line, i) => (
              <p key={i}>{line || '\u00A0'}</p>
            ))}
            {(msg.meta?.precedents_used ?? 0) > 0 && (
              <span className="precedent-count">
                Based on {msg.meta.precedents_used} precedent(s)
              </span>
            )}
          </div>
        </div>
      );
    }

    return null;
  }

  const firstStageIndex = messages.findIndex(m => m.role === 'stage');
  const beforeFirstStage = firstStageIndex === -1 ? messages : messages.slice(0, firstStageIndex);
  const fromFirstStage = firstStageIndex === -1 ? [] : messages.slice(firstStageIndex);

  return (
    <div className="chat-area">
      {loadedSession && (
        <>
          <div className="history-separator">— Loaded History —</div>
          {loadedSession.stage1 && (
            <Stage1Card key="h-stage1" data={loadedSession.stage1} />
          )}
          {loadedSession.stage2 && (
            <Stage2Card key="h-stage2" data={loadedSession.stage2} />
          )}
          {qaPairs.map((m, idx) => renderMessage({
            ...m,
            id: `h-${idx}`
          }))}
        </>
      )}


      {showLiveChat && (
        <>
          {beforeFirstStage.filter(m => m.role !== 'thought').map((msg) => renderMessage(msg))}
          {fromFirstStage.filter(m => m.role !== 'thought').map((msg) => renderMessage(msg))}
          {pipelineProgress && (
            <PipelineProgress progress={pipelineProgress} />
          )}
          {messages.filter(m => m.role === 'thought').map((msg) => renderMessage(msg))}
        </>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
