import React, { useState, useRef, useEffect } from 'react';
import { FiMessageCircle, FiSend, FiX, FiUser, FiCpu } from 'react-icons/fi';
import { sendChatQuery } from '../services/api';
import './ChatBot.css';

const ChatBot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      text: "Hello! I'm GridWatch Assistant. Ask me about district risks, consumer stats, anomalies, or revenue impact. Type **help** for all options.",
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = {
      type: 'user',
      text: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await sendChatQuery(input.trim());
      const botMessage = {
        type: 'bot',
        text: response.data.response,
        responseType: response.data.type,
        data: response.data.data,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        type: 'bot',
        text: "Sorry, I couldn't process that request. Please try again.",
        responseType: 'error',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const formatMessage = (text) => {
    // Convert markdown-style bold to HTML
    return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  };

  const quickQueries = [
    "Which district has highest risk?",
    "How many consumers?",
    "Total revenue at risk?",
    "Show anomaly types"
  ];

  return (
    <>
      {/* Chat Toggle Button */}
      <button 
        className={`chat-toggle ${isOpen ? 'hidden' : ''}`}
        onClick={() => setIsOpen(true)}
      >
        <FiMessageCircle />
        <span>Ask GridWatch</span>
      </button>

      {/* Chat Window */}
      <div className={`chat-window ${isOpen ? 'open' : ''}`}>
        <div className="chat-header">
          <div className="chat-title">
            <FiCpu className="bot-icon" />
            <span>GridWatch Assistant</span>
          </div>
          <button className="close-btn" onClick={() => setIsOpen(false)}>
            <FiX />
          </button>
        </div>

        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.type}`}>
              <div className="message-avatar">
                {msg.type === 'user' ? <FiUser /> : <FiCpu />}
              </div>
              <div className="message-content">
                <div 
                  className="message-text"
                  dangerouslySetInnerHTML={{ __html: formatMessage(msg.text) }}
                />
                <div className="message-time">
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="message bot">
              <div className="message-avatar"><FiCpu /></div>
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Queries */}
        {messages.length <= 2 && (
          <div className="quick-queries">
            {quickQueries.map((q, i) => (
              <button key={i} onClick={() => setInput(q)}>
                {q}
              </button>
            ))}
          </div>
        )}

        <div className="chat-input">
          <input
            type="text"
            placeholder="Ask about grid status..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
          />
          <button onClick={handleSend} disabled={isLoading || !input.trim()}>
            <FiSend />
          </button>
        </div>
      </div>
    </>
  );
};

export default ChatBot;
