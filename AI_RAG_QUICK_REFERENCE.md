# AuraLearn AI & RAG System - Quick Reference

## 🚀 Quick Start

### Essential Commands
```bash
# Setup
php artisan migrate
php artisan rag:create-sample-data
php artisan rag:test-system

# Health Check
curl http://localhost:8000/api/aurabot/health

# Test AuraBot
curl -X POST http://localhost:8000/api/aurabot/ask \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test_123","question":"How do I create HTML forms?"}'
```

## 🔧 Key Services

### AuraBot (Conversational AI)
- **File**: `app/Services/AuraBotRagService.php`
- **Purpose**: HTML learning assistant with contextual understanding
- **Features**: Unlimited questions, activity-specific sessions, refuses to give direct code

### AI Code Validation
- **File**: `app/Services/AiValidationService.php`
- **Purpose**: Intelligent code assessment with detailed feedback
- **Features**: Requirement analysis, educational feedback, progress tracking

### RAG System
- **File**: `app/Services/RagEmbeddingService.php`
- **Purpose**: Semantic content retrieval for enhanced AI responses
- **Features**: Vector search, document embeddings, intelligent chunking

### Nebius Client
- **File**: `app/Services/NebiusClient.php`
- **Purpose**: AI API integration with chat completions and embeddings
- **Features**: Mock responses for testing, connection management, error handling

## 🗄️ Database Tables

```sql
-- RAG Documents (with vector embeddings)
rag_documents (id, content, source, embedding, metadata)

-- Chatbot Sessions (activity-specific)
chatbot_sessions (session_id, user_id, attempt_count, progress_data)

-- Conversation History
chatbot_conversations (session_id, user_message, bot_response, created_at)

-- Embedding Cache
rag_embeddings_cache (content_hash, embedding, model)
```

## 🌐 API Endpoints

### AuraBot
```bash
POST /api/aurabot/ask                    # Ask question with context
GET  /api/aurabot/session-status         # Get session info
GET  /api/aurabot/conversation-history   # Get chat history
GET  /api/aurabot/health                 # System health check
```

### Activities with AI
```bash
POST /api/activities/{id}/submit         # Submit code for AI validation
GET  /api/activities/{id}/status         # Get submission status
```

## ⚙️ Configuration (.env)

```env
# AI Configuration
NEBIUS_API_KEY=your_api_key
NEBIUS_BASE_URL=https://api.studio.nebius.com/v1/
NEBIUS_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=BAAI/bge-multilingual-gemma2

# Vector Settings
VECTOR_DIM=1024

# RAG Settings
RAG_MAX_CHUNKS=5
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200

# AuraBot Settings
AURABOT_MAX_TOKENS=1000
AURABOT_ATTEMPT_LIMIT=999  # Unlimited
```

## 🧪 Testing Scripts

```bash
# Quick health check
./quick-test.bat

# Full AuraBot test
./test-aurabot-now.bat

# AI validation test
./test-ai-validation.bat

# Complete system test
./test-final.bat

# Offline system test
php artisan rag:test-offline
```

## 🚨 Troubleshooting

### Common Issues & Solutions

#### Nebius API Issues
```bash
# Check API key
grep NEBIUS_API_KEY .env

# Test connectivity
curl -H "Authorization: Bearer $NEBIUS_API_KEY" \
     https://api.studio.nebius.com/v1/models
```

#### pgvector Issues
```sql
-- Check extension
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Install if missing
CREATE EXTENSION IF NOT EXISTS vector;
```

#### Empty RAG Store
```bash
# Check document count
psql auralearn -c "SELECT COUNT(*) FROM rag_documents;"

# Add sample data
php artisan rag:create-sample-data
```

#### Session Problems
```bash
# Check sessions
psql auralearn -c "SELECT COUNT(*) FROM chatbot_sessions;"

# Clear old sessions
php artisan tinker
>>> ChatbotSession::where('created_at', '<', now()->subDays(1))->delete();
```

## 📊 Key Features

### ✅ Implemented Features
- **Contextual AI Responses**: Code + instructions + feedback awareness
- **Unlimited Questions**: No attempt restrictions
- **Activity-Specific Sessions**: Isolated conversation histories
- **AI Code Validation**: Intelligent assessment with educational feedback
- **Educational Guidance**: Refuses direct code, provides hints
- **Vector Search**: Semantic document retrieval
- **Fallback Systems**: Works even when external APIs fail
- **Comprehensive Testing**: Automated and manual test suites

### 🎯 AuraBot Capabilities
- Analyzes student HTML code
- Provides hints and guidance (no direct solutions)
- Understands activity instructions and requirements
- Considers previous submission feedback
- Offers encouragement and learning support
- Handles various question types appropriately

### 🔍 AI Validation Features
- Evaluates code against specific requirements
- Provides detailed technical analysis
- Generates concise, actionable feedback
- Calculates comprehensive scores
- Tracks learning progress
- Falls back to basic validation if AI unavailable

## 💡 Usage Examples

### Frontend Integration
```typescript
// Ask AuraBot
const response = await auraBotAPI.askQuestion(
  sessionId,
  "How do I add images to HTML?",
  htmlCode,
  activityInstructions,
  previousFeedback
);

// Submit for AI validation
const result = await fetch(`/api/activities/${id}/submit`, {
  method: 'POST',
  body: JSON.stringify({
    user_code: htmlCode,
    time_spent_minutes: 15
  })
});
```

### Backend Service Usage
```php
// Process AuraBot question
$result = $auraBotService->processUserQuestion(
    $sessionId,
    $question,
    $htmlContext,
    $instructionsContext,
    $feedbackContext
);

// Validate code with AI
$validation = $aiValidationService->validateCodeWithAi(
    $userCode,
    $instructions,
    $activityTitle
);
```

## 📈 Performance Tips

### Database Optimization
```sql
-- Essential indexes
CREATE INDEX CONCURRENTLY rag_documents_embedding_idx ON rag_documents 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX CONCURRENTLY chatbot_conversations_session_created_idx 
ON chatbot_conversations(session_id, created_at DESC);
```

### Caching Strategy
- **Embeddings**: Cache for 24 hours to reduce API calls
- **Sessions**: In-memory caching for active sessions
- **Documents**: Cache frequent searches

### Monitoring
```bash
# Check logs
tail -f storage/logs/laravel.log | grep "AuraBot"

# Monitor token usage
grep "tokens_used" storage/logs/laravel.log

# Check response times
grep "response_time_ms" storage/logs/laravel.log
```

---

## 📚 Related Files

- **Complete Documentation**: `AI_RAG_SYSTEM_COMPLETE_DOCUMENTATION.md`
- **System Overview**: `AURABOT_RAG_SYSTEM_DOCUMENTATION.md`
- **Deployment Guide**: `DEPLOYMENT_CHECKLIST.md`
- **Setup Instructions**: `RAG_SYSTEM_SETUP.md`

---

**Quick Reference Version**: 1.0  
**For Complete Details**: See `AI_RAG_SYSTEM_COMPLETE_DOCUMENTATION.md`
