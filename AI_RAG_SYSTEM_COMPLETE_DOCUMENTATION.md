# AuraLearn AI & RAG System - Complete Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [AI Components](#ai-components)
4. [RAG System](#rag-system)
5. [Database Schema](#database-schema)
6. [API Endpoints](#api-endpoints)
7. [Services Documentation](#services-documentation)
8. [Configuration](#configuration)
9. [Environment Setup](#environment-setup)
10. [Testing Framework](#testing-framework)
11. [Deployment](#deployment)
12. [Usage Examples](#usage-examples)
13. [Error Handling](#error-handling)
14. [Performance & Monitoring](#performance--monitoring)
15. [Security](#security)
16. [Troubleshooting](#troubleshooting)

---

## System Overview

The AuraLearn platform integrates advanced AI capabilities through three main systems:

### 🤖 **AuraBot** - Conversational HTML Learning Assistant
- **Purpose**: Interactive AI tutor for HTML/CSS learning
- **Features**: Context-aware responses, code analysis, educational guidance
- **Technology**: Nebius API (gpt-oss-20b model)
- **Capabilities**: Unlimited questions, activity-specific sessions, contextual understanding

### 🔍 **RAG System** - Retrieval-Augmented Generation
- **Purpose**: Enhanced AI responses using relevant educational content
- **Features**: Vector search, document embeddings, semantic retrieval
- **Technology**: PostgreSQL with pgvector, BAAI/bge-multilingual-gemma2 embeddings
- **Capabilities**: Intelligent content retrieval, contextual knowledge base

### ✅ **AI Code Validation** - Intelligent Assessment
- **Purpose**: Automated code evaluation and feedback
- **Features**: Requirement analysis, detailed feedback, score calculation
- **Technology**: Nebius AI with structured prompts
- **Capabilities**: Comprehensive validation, educational feedback, progress tracking

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                      │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │   Activity UI   │ │   AuraBot UI    │ │  Progress UI    ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/API Calls
┌─────────────────────▼───────────────────────────────────────┐
│                Backend (Laravel)                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │   Controllers   │ │    Services     │ │     Models      ││
│  │                 │ │                 │ │                 ││
│  │ • AuraBotCtrl   │ │ • AuraBotRAG    │ │ • ChatbotSess   ││
│  │ • ActivityCtrl  │ │ • NebiusClient  │ │ • Conversation  ││
│  │ • AdminCtrl     │ │ • AiValidation  │ │ • RagDocument   ││
│  │ • UserAuthCtrl  │ │ • RagEmbedding  │ │ • Activity      ││
│  │ • CourseCtrl    │ │ • TempDatabase  │ │ • User          ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   Data Layer                               │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │   PostgreSQL    │ │   File Storage  │ │  External APIs  ││
│  │                 │ │                 │ │                 ││
│  │ • Vector Store  │ │ • Temp Sessions │ │ • Nebius API    ││
│  │ • RAG Docs      │ │ • Activity Data │ │ • Embeddings    ││
│  │ • Conversations │ │ • Logs          │ │ • Chat Completn ││
│  │ • User Data     │ │                 │ │                 ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Component Relationships

```
AuraBot Frontend ←→ AuraBotController ←→ AuraBotRagService ←→ NebiusClient
                                      ↕                    ↕
Activity Frontend ←→ ActivityController ←→ AiValidationService ←→ Nebius API
                                      ↕
                              TemporaryDatabaseService
                                      ↕
                              PostgreSQL + pgvector
```

---

## AI Components

### 1. **Nebius AI Integration**

**File**: `app/Services/NebiusClient.php`

The primary AI service connecting to Nebius API for chat completions and embeddings.

#### **Features**:
- Chat completions using `gpt-oss-20b` model
- Text embeddings using `BAAI/bge-multilingual-gemma2`
- Mock responses for development/testing
- Connection testing and health checks
- Error handling and fallback mechanisms

#### **Key Methods**:

```php
// Chat completion with context
public function createChatCompletion(array $messages, array $options = []): array

// Generate text embeddings
public function createEmbedding(string $text, string $model = 'BAAI/bge-multilingual-gemma2'): array

// Test API connectivity
public function testConnection(): array
```

#### **Configuration**:
```env
NEBIUS_API_KEY=your_api_key_here
NEBIUS_BASE_URL=https://api.studio.nebius.com/v1/
NEBIUS_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=BAAI/bge-multilingual-gemma2
VECTOR_DIM=1024
```

### 2. **AuraBot RAG Service**

**File**: `app/Services/AuraBotRagService.php`

Core orchestration service combining retrieval and generation for educational AI responses.

#### **Features**:
- **Context-Aware Responses**: Analyzes user code, instructions, and feedback
- **Educational Focus**: Refuses to provide direct code, offers hints instead
- **Session Management**: Activity-specific conversation histories
- **Unlimited Questions**: No attempt limits, encouraging exploration
- **Semantic Search**: Retrieves relevant educational content

#### **Core Workflow**:

```php
public function processUserQuestion(
    string $sessionId,
    string $question,
    ?string $htmlContext = null,
    ?string $instructionsContext = null,
    ?string $feedbackContext = null,
    ?int $userId = null
): array
```

**Process Flow**:
1. **Context Building**: Combines user code, activity instructions, and feedback
2. **Retrieval**: Searches RAG documents for relevant content
3. **Conversation History**: Includes recent chat context
4. **AI Generation**: Creates educational response using Nebius API
5. **Session Update**: Stores conversation and updates progress

#### **Response Types**:
- **Code Analysis**: When user asks about their HTML
- **General Help**: HTML concepts and guidance
- **Completion Assistance**: Help finishing activities
- **Error Guidance**: Specific fixes and improvements

### 3. **AI Code Validation**

**File**: `app/Services/AiValidationService.php`

Intelligent assessment system for student code submissions.

#### **Features**:
- **Requirement Analysis**: Evaluates code against specific instructions
- **Technical Validation**: Checks HTML structure, syntax, semantics
- **Educational Feedback**: Concise, actionable improvement suggestions
- **Scoring System**: Detailed breakdown with explanations
- **Fallback Validation**: Basic checks when AI is unavailable

#### **Validation Process**:

```php
public function validateCodeWithAi($userCode, $instructions, $activityTitle, $activityDescription = null)
```

**Assessment Categories**:
1. **Overall Score** (0-100): Based on requirement completion
2. **Completion Status**: `passed` (≥80%), `partial` (60-79%), `failed` (<60%)
3. **Requirements Analysis**: Individual requirement evaluation
4. **Technical Validation**: Structure, syntax, semantics, accessibility
5. **Educational Feedback**: Brief, focused improvement suggestions

#### **Response Structure**:
```json
{
  "ai_powered": true,
  "overall_score": 85,
  "completion_status": "passed",
  "is_completed": true,
  "requirements_analysis": [...],
  "technical_validation": {...},
  "detailed_feedback": "Brief feedback text",
  "suggestions": ["Fix this", "Improve that"],
  "positive_aspects": ["Good structure"],
  "areas_for_improvement": ["Add alt attributes"]
}
```

---

## RAG System

### **Document Storage & Retrieval**

**File**: `app/Services/RagEmbeddingService.php`

#### **Features**:
- **Vector Embeddings**: 1024-dimensional vectors using BAAI model
- **Semantic Search**: Cosine similarity for document retrieval
- **Document Chunking**: Intelligent text segmentation
- **Caching System**: Reduces API calls with embedding cache
- **pgvector Integration**: PostgreSQL extension for vector operations

#### **Document Processing**:

```php
// Ingest educational content
public function ingestDocument(
    string $content,
    string $source,
    string $documentType = 'text',
    array $metadata = []
): int

// Search for relevant content
public function searchRelevantDocuments(
    string $query,
    int $limit = null,
    float $threshold = 0.7,
    array $documentTypes = []
): Collection
```

#### **Text Chunking Strategy**:
- **Chunk Size**: Configurable (default: 1000 characters)
- **Overlap**: Prevents context loss (default: 200 characters)
- **Smart Splitting**: Preserves sentence boundaries
- **Metadata Preservation**: Maintains source and context information

### **Vector Search Process**:

1. **Query Embedding**: Convert user question to vector
2. **Similarity Search**: Find relevant document chunks
3. **Threshold Filtering**: Remove low-relevance results
4. **Context Assembly**: Combine retrieved chunks
5. **Response Generation**: Use context for AI completion

---

## Database Schema

### **RAG Documents Table**

```sql
CREATE TABLE rag_documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    source VARCHAR(500) NOT NULL,
    document_type VARCHAR(100) DEFAULT 'text',
    metadata JSONB,
    embedding vector(1024), -- pgvector extension
    embedding_model VARCHAR(100) DEFAULT 'BAAI/bge-multilingual-gemma2',
    embedding_dimensions INTEGER DEFAULT 1024,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector similarity index
CREATE INDEX rag_documents_embedding_idx ON rag_documents 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### **Chatbot Sessions Table**

```sql
CREATE TABLE chatbot_sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id BIGINT,
    attempt_count INTEGER DEFAULT 0,
    is_blocked BOOLEAN DEFAULT FALSE,
    last_question_at TIMESTAMP,
    progress_data JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### **Chatbot Conversations Table**

```sql
CREATE TABLE chatbot_conversations (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    user_id BIGINT,
    user_message TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    context_data JSONB,
    tokens_used INTEGER DEFAULT 0,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (session_id) REFERENCES chatbot_sessions(session_id) ON DELETE CASCADE
);
```

### **RAG Embeddings Cache Table**

```sql
CREATE TABLE rag_embeddings_cache (
    id BIGSERIAL PRIMARY KEY,
    content_hash VARCHAR(64) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),
    model VARCHAR(100) DEFAULT 'BAAI/bge-multilingual-gemma2',
    dimensions INTEGER DEFAULT 1024,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## API Endpoints

### **AuraBot Endpoints**

#### **POST /api/aurabot/ask**
Ask a question to AuraBot with full context.

**Request**:
```json
{
  "session_id": "activity_123_user_456",
  "question": "How do I add an image to my HTML?",
  "html_context": "<!DOCTYPE html><html>...",
  "instructions_context": "Create a webpage with...",
  "feedback_context": "Previous feedback...",
  "user_id": 456
}
```

**Response**:
```json
{
  "success": true,
  "response": "To add an image, use the <img> tag...",
  "conversation_id": 789,
  "tokens_used": 150,
  "remaining_attempts": 999
}
```

#### **GET /api/aurabot/session-status**
Get session information and status.

**Parameters**: `session_id`

**Response**:
```json
{
  "exists": true,
  "can_ask": true,
  "attempt_count": 5,
  "remaining_attempts": 999,
  "last_question_at": "2025-01-01T12:00:00Z"
}
```

#### **GET /api/aurabot/conversation-history**
Retrieve conversation history for a session.

**Parameters**: `session_id`, `limit` (optional, default: 20)

**Response**:
```json
{
  "success": true,
  "messages": [
    {
      "id": 1,
      "user_message": "How do I create a form?",
      "bot_response": "To create a form...",
      "created_at": "2025-01-01T12:00:00Z"
    }
  ],
  "total_count": 15
}
```

#### **GET /api/aurabot/health**
System health check.

**Response**:
```json
{
  "success": true,
  "status": "healthy",
  "database": "connected",
  "nebius_api": "connected",
  "timestamp": "2025-01-01T12:00:00Z"
}
```

### **Activity Endpoints with AI**

#### **POST /api/activities/{id}/submit**
Submit activity with AI validation.

**Request**:
```json
{
  "user_code": "<!DOCTYPE html>...",
  "time_spent_minutes": 15
}
```

**Response**:
```json
{
  "success": true,
  "submission_id": 123,
  "score": 85,
  "is_completed": true,
  "completion_status": "passed",
  "attempt_number": 2,
  "feedback": "Great job! Your HTML structure...",
  "validation_summary": {
    "overall": {
      "passed": 8,
      "total": 10,
      "percentage": 80
    }
  },
  "instruction_progress": {
    "completed": 4,
    "total": 5,
    "percentage": 80
  },
  "ai_powered": true
}
```

#### **GET /api/activities/{id}/status**
Get submission status and progress.

**Response**:
```json
{
  "activity_id": 123,
  "total_attempts": 2,
  "max_attempts": null,
  "attempts_remaining": null,
  "is_completed": false,
  "best_score": 75,
  "latest_submission": {
    "id": 456,
    "score": 75,
    "feedback": "Good progress..."
  }
}
```

---

## Services Documentation

### **AuraBotRagService**

**Primary Functions**:

```php
// Main question processing
public function processUserQuestion(
    string $sessionId,
    string $question,
    ?string $htmlContext = null,
    ?string $instructionsContext = null,
    ?string $feedbackContext = null,
    ?int $userId = null
): array

// Session management
public function getSessionStatus(string $sessionId): array
public function getConversationHistory(string $sessionId, int $limit = 20): array
public function resetSession(string $sessionId): bool
```

**Internal Methods**:

```php
// Context building
private function buildEditorContext(?string $htmlContext, ?string $instructionsContext, ?string $feedbackContext): string
private function buildConversationContext(Collection $history): string
private function buildRetrievedContext(Collection $documents): string

// AI prompt construction
private function buildSystemPrompt(int $attemptNumber): string
private function buildUserPrompt(string $question, string $retrievedContext, string $conversationContext, string $editorContext): string

// Code analysis
private function analyzeStudentHtml(string $editorContext): string
```

### **NebiusClient**

**Primary Functions**:

```php
// AI interactions
public function createChatCompletion(array $messages, array $options = []): array
public function createEmbedding(string $text, string $model = 'BAAI/bge-multilingual-gemma2'): array

// System utilities
public function testConnection(): array
```

**Configuration Options**:

```php
// Chat completion options
$options = [
    'max_tokens' => 1000,
    'temperature' => 0.7,
    'top_p' => 0.9
];
```

### **RagEmbeddingService**

**Document Management**:

```php
// Ingest content
public function ingestDocument(string $content, string $source, string $documentType = 'text', array $metadata = []): int

// Search documents
public function searchRelevantDocuments(string $query, int $limit = null, float $threshold = 0.7, array $documentTypes = []): Collection

// Text processing
public function chunkText(string $text, int $chunkSize = null, int $overlap = null): array
public function generateEmbedding(string $text, string $model = 'BAAI/bge-multilingual-gemma2'): array
```

### **AiValidationService**

**Validation Functions**:

```php
// Main validation
public function validateCodeWithAi($userCode, $instructions, $activityTitle, $activityDescription = null)

// Feedback generation
public function generateEducationalFeedback($validationResult)
public function generateQuickFeedback($userCode, $requirements, $score)
```

### **TemporaryDatabaseService**

**Session Data Management**:

```php
// Submission tracking
public function storeSubmission($data)
public function getSubmissionStatus($userId, $activityId)

// Data management
public function logActivity($data)
public function clearUserActivityData($userId, $activityId)
public function clearAllData()
```

---

## Configuration

### **Environment Variables**

```env
# Nebius AI Configuration
NEBIUS_API_KEY=your_nebius_api_key
NEBIUS_BASE_URL=https://api.studio.nebius.com/v1/
NEBIUS_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=BAAI/bge-multilingual-gemma2

# Vector Configuration
VECTOR_DIM=1024

# RAG Configuration
RAG_MAX_CHUNKS=5
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200

# AuraBot Configuration
AURABOT_MAX_TOKENS=1000
AURABOT_ATTEMPT_LIMIT=999

# Database Configuration
DB_CONNECTION=pgsql
DB_HOST=localhost
DB_PORT=5432
DB_DATABASE=auralearn
DB_USERNAME=your_username
DB_PASSWORD=your_password

# Application Configuration
APP_NAME="AuraLearn"
APP_ENV=production
APP_DEBUG=false
APP_URL=https://your-domain.com
```

### **Service Provider Registration**

**File**: `app/Providers/RagServiceProvider.php`

```php
public function register(): void
{
    // Core AI services
    $this->app->singleton(NebiusClient::class);
    $this->app->singleton(RagEmbeddingService::class);
    $this->app->singleton(AuraBotRagService::class);
    $this->app->singleton(AiValidationService::class);
    
    // Utility services
    $this->app->singleton(AuraBotApiService::class);
    $this->app->singleton(TemporaryDatabaseService::class);
}

public function boot(): void
{
    // Register Artisan commands
    if ($this->app->runningInConsole()) {
        $this->commands([
            IngestRagDocuments::class,
            IngestCourseContent::class,
            CreateSampleRagData::class,
            TestRagSystem::class,
            TestSystemWithoutAPIs::class,
        ]);
    }
}
```

---

## Environment Setup

### **1. Database Setup**

```bash
# Create PostgreSQL database
createdb auralearn

# Install pgvector extension (if available)
psql auralearn -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run migrations
php artisan migrate
```

### **2. Dependencies Installation**

```bash
# PHP dependencies
composer install

# Install required packages
composer require openai-php/client pgvector/pgvector guzzlehttp/guzzle
```

### **3. Service Registration**

**File**: `bootstrap/providers.php`

```php
return [
    // ... other providers
    App\Providers\RagServiceProvider::class,
];
```

### **4. Initial Data Setup**

```bash
# Create sample RAG documents
php artisan rag:create-sample-data

# Ingest course content
php artisan rag:ingest-course-content

# Test system
php artisan rag:test-system
```

---

## Testing Framework

### **Automated Testing Scripts**

#### **Health Check**
```bash
# File: quick-test.bat
curl http://localhost:8000/api/aurabot/health
```

#### **AuraBot Functionality**
```bash
# File: test-aurabot-now.bat
# Tests session creation, question asking, conversation history
```

#### **AI Validation**
```bash
# File: test-ai-validation.bat
# Tests code submission with AI validation
```

#### **Comprehensive System Test**
```bash
# File: test-final.bat
# Full system integration test
```

### **Unit Testing Commands**

```bash
# Test RAG system without external APIs
php artisan rag:test-offline

# Test specific components
php artisan test --filter=AuraBotTest
php artisan test --filter=RagSystemTest
php artisan test --filter=AiValidationTest
```

### **Manual Testing Procedures**

#### **1. AuraBot Testing**

```powershell
# Test basic question
$response = Invoke-RestMethod -Uri "http://localhost:8000/api/aurabot/ask" -Method POST -Headers @{"Content-Type"="application/json"} -Body (@{
    session_id = "test_session_123"
    question = "How do I create an HTML form?"
    html_context = "<!DOCTYPE html><html><head><title>Test</title></head><body></body></html>"
    instructions_context = "Create a contact form with name, email, and message fields"
} | ConvertTo-Json)
```

#### **2. AI Validation Testing**

```powershell
# Test code submission
$submission = Invoke-RestMethod -Uri "http://localhost:8000/api/activities/1/submit" -Method POST -Headers @{"Content-Type"="application/json"} -Body (@{
    user_code = "<!DOCTYPE html><html><head><title>My Page</title></head><body><h1>Hello World</h1></body></html>"
    time_spent_minutes = 10
} | ConvertTo-Json)
```

#### **3. Session Management Testing**

```powershell
# Check session status
$status = Invoke-RestMethod -Uri "http://localhost:8000/api/aurabot/session-status?session_id=test_session_123" -Method GET

# Get conversation history
$history = Invoke-RestMethod -Uri "http://localhost:8000/api/aurabot/conversation-history?session_id=test_session_123&limit=10" -Method GET
```

---

## Deployment

### **Production Setup**

#### **1. Environment Configuration**

```env
# Production settings
APP_ENV=production
APP_DEBUG=false
APP_URL=https://auralearn.com

# Optimized database settings
DB_CONNECTION=pgsql
DB_HOST=production_db_host
DB_PORT=5432
DB_DATABASE=auralearn_prod

# Production API keys
NEBIUS_API_KEY=production_nebius_key

# Performance settings
RAG_MAX_CHUNKS=10
AURABOT_MAX_TOKENS=1500
```

#### **2. Database Optimization**

```sql
-- Create optimized indexes
CREATE INDEX CONCURRENTLY chatbot_sessions_session_id_idx ON chatbot_sessions(session_id);
CREATE INDEX CONCURRENTLY chatbot_conversations_session_created_idx ON chatbot_conversations(session_id, created_at);
CREATE INDEX CONCURRENTLY rag_documents_type_idx ON rag_documents(document_type);

-- Vacuum and analyze
VACUUM ANALYZE rag_documents;
VACUUM ANALYZE chatbot_sessions;
VACUUM ANALYZE chatbot_conversations;
```

#### **3. Caching Setup**

```php
// Config: config/cache.php
'stores' => [
    'redis' => [
        'driver' => 'redis',
        'connection' => 'cache',
        'lock_connection' => 'default',
    ],
],

// RAG embedding cache
'embedding_cache_ttl' => 86400, // 24 hours
```

#### **4. Queue Configuration**

```php
// Config: config/queue.php
'connections' => [
    'redis' => [
        'driver' => 'redis',
        'connection' => 'default',
        'queue' => env('REDIS_QUEUE', 'default'),
    ],
],
```

### **Production Checklist**

- [ ] Environment variables configured
- [ ] Database migrations run
- [ ] pgvector extension installed
- [ ] RAG documents ingested
- [ ] Indexes created and optimized
- [ ] Cache configured (Redis recommended)
- [ ] Queue workers running
- [ ] SSL certificates installed
- [ ] Monitoring configured
- [ ] Backup procedures in place
- [ ] Rate limiting configured
- [ ] Security headers configured

---

## Usage Examples

### **Frontend Integration**

#### **AuraBot Component Integration**

```typescript
// lib/aurabot-api.ts
class AuraBotAPI {
  async askQuestion(
    sessionId: string,
    question: string,
    htmlContext?: string,
    instructionsContext?: string,
    feedbackContext?: string,
    userId?: number
  ): Promise<AuraBotResponse> {
    const response = await fetch('/api/aurabot/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        question,
        html_context: htmlContext || null,
        instructions_context: instructionsContext || null,
        feedback_context: feedbackContext || null,
        user_id: userId || null
      })
    });
    
    return response.json();
  }
}
```

#### **Activity Submission**

```typescript
// Submit code for AI validation
const submitCode = async (code: string, activityId: number) => {
  const response = await fetch(`/api/activities/${activityId}/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_code: code,
      time_spent_minutes: timeSpent
    })
  });
  
  const result = await response.json();
  
  if (result.success) {
    // Handle successful submission
    console.log('Score:', result.score);
    console.log('Feedback:', result.feedback);
    console.log('Completed:', result.is_completed);
  }
};
```

### **Backend Service Usage**

#### **Direct Service Usage**

```php
// In a controller or service
class ExampleController extends Controller
{
    public function __construct(
        private AuraBotRagService $auraBotService,
        private AiValidationService $aiValidation
    ) {}
    
    public function askAuraBot(Request $request)
    {
        $result = $this->auraBotService->processUserQuestion(
            sessionId: $request->session_id,
            question: $request->question,
            htmlContext: $request->html_context,
            instructionsContext: $request->instructions_context,
            feedbackContext: $request->feedback_context
        );
        
        return response()->json($result);
    }
    
    public function validateCode(Request $request)
    {
        $validation = $this->aiValidation->validateCodeWithAi(
            userCode: $request->code,
            instructions: $request->instructions,
            activityTitle: $request->title
        );
        
        return response()->json($validation);
    }
}
```

#### **RAG Document Management**

```php
// Ingest educational content
$ragService = app(RagEmbeddingService::class);

$documentId = $ragService->ingestDocument(
    content: "HTML forms allow users to input data...",
    source: "html_forms_tutorial.md",
    documentType: "tutorial",
    metadata: [
        'topic' => 'HTML Forms',
        'difficulty' => 'beginner',
        'tags' => ['html', 'forms', 'input']
    ]
);

// Search for relevant content
$relevantDocs = $ragService->searchRelevantDocuments(
    query: "How do I create a form?",
    limit: 5,
    threshold: 0.7
);
```

### **Command Line Usage**

```bash
# Create and populate RAG documents
php artisan rag:create-sample-data
php artisan rag:ingest-course-content

# Test system components
php artisan rag:test-system
php artisan rag:test-offline

# Clear temporary data
curl -X DELETE http://localhost:8000/api/activities/clear-temporary-data

# Health check
curl http://localhost:8000/api/aurabot/health
```

---

## Error Handling

### **API Error Responses**

#### **Standard Error Format**

```json
{
  "success": false,
  "error": "Error description",
  "details": {
    "field": ["Validation error message"]
  },
  "code": "ERROR_CODE",
  "timestamp": "2025-01-01T12:00:00Z"
}
```

#### **AuraBot Errors**

```json
// Rate limiting (not currently implemented)
{
  "success": false,
  "error": "Rate limit exceeded",
  "retry_after": 60
}

// Invalid session
{
  "success": false,
  "error": "Invalid session ID",
  "code": "INVALID_SESSION"
}

// AI service unavailable
{
  "success": false,
  "error": "AI service temporarily unavailable",
  "fallback_used": true
}
```

### **Service Error Handling**

#### **NebiusClient Fallbacks**

```php
// Automatic fallback to mock responses
if (!$this->apiKey) {
    Log::info('Using mock response - Nebius API key not configured');
    return $this->createMockChatCompletion($messages, $options);
}

// Connection error handling
try {
    $response = Http::withHeaders($headers)->post($url, $payload);
} catch (\Exception $e) {
    Log::error('Nebius API error', ['error' => $e->getMessage()]);
    return $this->createMockChatCompletion($messages, $options);
}
```

#### **Database Error Handling**

```php
// pgvector extension fallback
try {
    DB::statement('CREATE EXTENSION IF NOT EXISTS vector');
} catch (\Exception $e) {
    Log::warning('pgvector extension not available, using JSON fallback');
    // Continue with JSON-based similarity
}
```

#### **Validation Error Handling**

```php
// AI validation fallback
try {
    $aiValidationResult = $this->aiValidationService->validateCodeWithAi(...);
} catch (\Exception $aiError) {
    Log::error('AI validation failed, using fallback', [
        'error' => $aiError->getMessage()
    ]);
    
    // Use basic validation
    $aiValidationResult = $this->getFallbackValidation($userCode);
}
```

### **Logging Strategy**

#### **Log Levels and Categories**

```php
// Info logs
Log::info('AuraBot question processed', [
    'session_id' => $sessionId,
    'success' => $result['success'],
    'tokens_used' => $result['tokens_used'] ?? 0
]);

// Warning logs
Log::warning('Using mock response - Nebius API key not configured');

// Error logs
Log::error('AI validation failed', [
    'error' => $e->getMessage(),
    'activity_id' => $activityId,
    'trace' => $e->getTraceAsString()
]);
```

#### **Log Analysis**

```bash
# Monitor AuraBot usage
tail -f storage/logs/laravel.log | grep "AuraBot"

# Check AI validation errors
grep "AI validation failed" storage/logs/laravel.log

# Monitor API errors
grep "Nebius API error" storage/logs/laravel.log
```

---

## Performance & Monitoring

### **Performance Optimization**

#### **Database Optimization**

```sql
-- Optimize vector searches
CREATE INDEX CONCURRENTLY rag_documents_embedding_idx ON rag_documents 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Optimize conversation queries
CREATE INDEX CONCURRENTLY chatbot_conversations_session_created_idx 
ON chatbot_conversations(session_id, created_at DESC);

-- Optimize session lookups
CREATE UNIQUE INDEX CONCURRENTLY chatbot_sessions_session_id_idx 
ON chatbot_sessions(session_id);
```

#### **Caching Strategy**

```php
// Cache embeddings
class RagEmbeddingService
{
    public function generateEmbedding(string $text, string $model = 'BAAI/bge-multilingual-gemma2'): array
    {
        $hash = hash('sha256', $text . $model);
        
        // Check cache first
        $cached = Cache::get("embedding:{$hash}");
        if ($cached) {
            return $cached;
        }
        
        // Generate and cache
        $embedding = $this->nebiusClient->createEmbedding($text, $model);
        Cache::put("embedding:{$hash}", $embedding, 86400); // 24 hours
        
        return $embedding;
    }
}
```

#### **Query Optimization**

```php
// Optimize conversation history retrieval
public function getConversationHistory(string $sessionId, int $limit = 20): Collection
{
    return ChatbotConversation::where('session_id', $sessionId)
        ->orderBy('created_at', 'desc')
        ->limit($limit)
        ->get(['user_message', 'bot_response', 'created_at'])
        ->reverse()
        ->values();
}

// Optimize RAG document search
public function searchRelevantDocuments(string $query, int $limit = null, float $threshold = 0.7): Collection
{
    $queryEmbedding = $this->generateEmbedding($query);
    
    return DB::table('rag_documents')
        ->select('content', 'source', 'metadata')
        ->selectRaw('1 - (embedding <=> ?) as similarity', [$queryEmbedding['embedding']])
        ->whereRaw('1 - (embedding <=> ?) > ?', [$queryEmbedding['embedding'], $threshold])
        ->orderByDesc('similarity')
        ->limit($limit ?? config('rag.max_chunks', 5))
        ->get();
}
```

### **Monitoring & Metrics**

#### **Health Check Endpoint**

```php
public function healthCheck(): JsonResponse
{
    $health = [
        'status' => 'healthy',
        'database' => 'connected',
        'nebius_api' => 'connected',
        'timestamp' => now()->toISOString()
    ];
    
    try {
        // Test database
        DB::connection()->getPdo();
        
        // Test Nebius API
        $nebiusClient = app(NebiusClient::class);
        $apiTest = $nebiusClient->testConnection();
        $health['nebius_api'] = $apiTest['success'] ? 'connected' : 'error';
        
        if (!$apiTest['success']) {
            $health['status'] = 'degraded';
            $health['nebius_error'] = $apiTest['error'];
        }
        
    } catch (\Exception $e) {
        $health['status'] = 'unhealthy';
        $health['error'] = $e->getMessage();
        return response()->json($health, 500);
    }
    
    return response()->json($health);
}
```

#### **Performance Metrics**

```php
// Track response times
$startTime = microtime(true);
$result = $this->auraBotService->processUserQuestion(...);
$responseTime = (microtime(true) - $startTime) * 1000; // milliseconds

Log::info('AuraBot response time', [
    'session_id' => $sessionId,
    'response_time_ms' => $responseTime,
    'tokens_used' => $result['tokens_used'] ?? 0
]);

// Track token usage
public function trackTokenUsage(string $sessionId, int $tokensUsed): void
{
    Cache::increment("tokens:daily:" . now()->format('Y-m-d'), $tokensUsed);
    Cache::increment("tokens:session:" . $sessionId, $tokensUsed);
}
```

#### **Usage Analytics**

```php
// Track user interactions
class AnalyticsService
{
    public function trackAuraBotUsage(string $sessionId, string $question, array $response): void
    {
        DB::table('analytics_events')->insert([
            'event_type' => 'aurabot_question',
            'session_id' => $sessionId,
            'data' => json_encode([
                'question_length' => strlen($question),
                'response_length' => strlen($response['response'] ?? ''),
                'tokens_used' => $response['tokens_used'] ?? 0,
                'success' => $response['success'] ?? false
            ]),
            'created_at' => now()
        ]);
    }
    
    public function trackCodeSubmission(int $activityId, array $validation): void
    {
        DB::table('analytics_events')->insert([
            'event_type' => 'code_submission',
            'activity_id' => $activityId,
            'data' => json_encode([
                'score' => $validation['overall_score'],
                'is_completed' => $validation['is_completed'],
                'ai_powered' => $validation['ai_powered'],
                'attempt_number' => $validation['attempt_number'] ?? 1
            ]),
            'created_at' => now()
        ]);
    }
}
```

---

## Security

### **API Security**

#### **Input Validation**

```php
// AuraBotController validation
public function askQuestion(Request $request): JsonResponse
{
    $validator = Validator::make($request->all(), [
        'session_id' => 'required|string|max:255|regex:/^[a-zA-Z0-9_-]+$/',
        'question' => 'required|string|max:2000',
        'html_context' => 'nullable|string|max:10000',
        'instructions_context' => 'nullable|string|max:5000',
        'feedback_context' => 'nullable|string|max:5000',
        'user_id' => 'nullable|integer|min:1'
    ]);
    
    if ($validator->fails()) {
        return response()->json([
            'success' => false,
            'error' => 'Invalid input data',
            'details' => $validator->errors()
        ], 422);
    }
}
```

#### **Rate Limiting**

```php
// Future implementation - rate limiting
class AuraBotRateLimiter
{
    public function canMakeRequest(string $sessionId): bool
    {
        $key = "aurabot_requests:{$sessionId}";
        $requests = Cache::get($key, 0);
        
        // Allow 60 requests per hour
        if ($requests >= 60) {
            return false;
        }
        
        Cache::put($key, $requests + 1, 3600);
        return true;
    }
}
```

#### **Content Sanitization**

```php
// Sanitize user input
class SecurityService
{
    public function sanitizeHtmlContext(string $html): string
    {
        // Remove potentially dangerous content
        $html = preg_replace('/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/mi', '', $html);
        $html = preg_replace('/on\w+="[^"]*"/i', '', $html);
        $html = preg_replace('/javascript:/i', '', $html);
        
        return $html;
    }
    
    public function sanitizeUserQuestion(string $question): string
    {
        // Remove HTML tags and limit length
        $question = strip_tags($question);
        $question = substr($question, 0, 2000);
        
        return trim($question);
    }
}
```

### **Data Protection**

#### **Sensitive Data Handling**

```php
// Encrypt sensitive session data
class ChatbotSession extends Model
{
    protected $casts = [
        'is_blocked' => 'boolean',
        'progress_data' => 'encrypted:array',
        'metadata' => 'encrypted:array'
    ];
}

// Anonymize user data in logs
Log::info('AuraBot question processed', [
    'session_id' => hash('sha256', $sessionId), // Hash instead of raw ID
    'question_length' => strlen($question),
    'success' => $result['success']
]);
```

#### **Database Security**

```sql
-- Create restricted database user
CREATE USER auralearn_app WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE auralearn TO auralearn_app;
GRANT USAGE ON SCHEMA public TO auralearn_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO auralearn_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO auralearn_app;

-- Row-level security (future implementation)
ALTER TABLE chatbot_sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY user_sessions ON chatbot_sessions FOR ALL TO auralearn_app 
USING (user_id = current_setting('app.current_user_id')::integer);
```

---

## Troubleshooting

### **Common Issues**

#### **1. Nebius API Connection Issues**

**Symptoms**: 
- Health check shows `nebius_api: error`
- Mock responses being used instead of real AI

**Solutions**:
```bash
# Check API key configuration
grep NEBIUS_API_KEY .env

# Test API connectivity
curl -H "Authorization: Bearer $NEBIUS_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.studio.nebius.com/v1/models

# Check logs
tail -f storage/logs/laravel.log | grep "Nebius API"
```

**Common Fixes**:
- Verify `NEBIUS_API_KEY` is correctly set
- Check network connectivity to `api.studio.nebius.com`
- Verify API key has proper permissions
- Check rate limits on Nebius account

#### **2. pgvector Extension Issues**

**Symptoms**:
- Database migration errors mentioning `vector`
- RAG searches returning empty results

**Solutions**:
```bash
# Check if pgvector is installed
psql auralearn -c "SELECT * FROM pg_extension WHERE extname = 'vector';"

# Install pgvector extension
psql auralearn -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify vector operations work
psql auralearn -c "SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector;"
```

**Fallback Mode**:
If pgvector is not available, the system automatically falls back to JSON-based similarity calculations.

#### **3. Empty RAG Document Store**

**Symptoms**:
- AuraBot responses are generic
- No relevant content found in searches

**Solutions**:
```bash
# Check document count
psql auralearn -c "SELECT COUNT(*) FROM rag_documents;"

# Ingest sample data
php artisan rag:create-sample-data

# Ingest course content
php artisan rag:ingest-course-content

# Test RAG system
php artisan rag:test-system
```

#### **4. Session Management Issues**

**Symptoms**:
- Sessions not persisting across requests
- Conversation history lost

**Solutions**:
```bash
# Check session table
psql auralearn -c "SELECT COUNT(*) FROM chatbot_sessions;"

# Clear corrupted sessions
php artisan tinker
>>> ChatbotSession::where('created_at', '<', now()->subDays(1))->delete();

# Test session creation
curl -X POST http://localhost:8000/api/aurabot/ask \
     -H "Content-Type: application/json" \
     -d '{"session_id":"test123","question":"Hello"}'
```

#### **5. AI Validation Not Working**

**Symptoms**:
- Submissions always use fallback validation
- `ai_powered: false` in responses

**Solutions**:
```bash
# Test AI validation service directly
php artisan tinker
>>> $service = app(\App\Services\AiValidationService::class);
>>> $result = $service->validateCodeWithAi('<html></html>', ['Create HTML page'], 'Test');
>>> var_dump($result);

# Check logs for AI validation errors
grep "AI validation failed" storage/logs/laravel.log
```

### **Performance Issues**

#### **Slow RAG Searches**

**Diagnosis**:
```sql
-- Check query performance
EXPLAIN ANALYZE SELECT content FROM rag_documents 
WHERE 1 - (embedding <=> '[0.1,0.2,...]'::vector) > 0.7 
ORDER BY embedding <=> '[0.1,0.2,...]'::vector LIMIT 5;

-- Check index usage
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'rag_documents';
```

**Solutions**:
```sql
-- Create missing indexes
CREATE INDEX CONCURRENTLY rag_documents_embedding_idx ON rag_documents 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Update table statistics
ANALYZE rag_documents;
```

#### **High Memory Usage**

**Diagnosis**:
```bash
# Check PHP memory usage
grep memory_limit /etc/php/8.2/fpm/php.ini

# Monitor process memory
ps aux | grep php
```

**Solutions**:
```php
// Optimize embedding caching
'embedding_cache_size' => 1000, // Limit cache size
'embedding_ttl' => 86400, // 24 hours

// Use chunked processing for large documents
public function ingestLargeDocument(string $content): void
{
    $chunks = $this->chunkText($content, 500, 50);
    
    foreach (array_chunk($chunks, 10) as $batch) {
        $this->processBatch($batch);
        gc_collect_cycles(); // Force garbage collection
    }
}
```

### **Debugging Tools**

#### **Artisan Commands**

```bash
# Test entire system
php artisan rag:test-system

# Test without external APIs
php artisan rag:test-offline

# Clear all temporary data
curl -X DELETE http://localhost:8000/api/activities/clear-temporary-data
```

#### **Log Analysis**

```bash
# Monitor real-time logs
tail -f storage/logs/laravel.log

# Filter specific components
grep "AuraBot" storage/logs/laravel.log
grep "AI validation" storage/logs/laravel.log
grep "Nebius API" storage/logs/laravel.log

# Count errors by type
grep "ERROR" storage/logs/laravel.log | cut -d' ' -f4- | sort | uniq -c | sort -nr
```

#### **Database Debugging**

```sql
-- Check recent sessions
SELECT session_id, attempt_count, last_question_at 
FROM chatbot_sessions 
ORDER BY updated_at DESC LIMIT 10;

-- Check conversation activity
SELECT session_id, COUNT(*) as message_count, MAX(created_at) as last_message
FROM chatbot_conversations 
GROUP BY session_id 
ORDER BY last_message DESC LIMIT 10;

-- Check RAG document distribution
SELECT document_type, COUNT(*) as count, AVG(LENGTH(content)) as avg_length
FROM rag_documents 
GROUP BY document_type;
```

---

## Conclusion

This comprehensive documentation covers the complete AI and RAG system integration in AuraLearn. The system provides:

### **Key Features Delivered**:
- ✅ **Intelligent Tutoring**: AuraBot provides contextual HTML learning assistance
- ✅ **Advanced Code Validation**: AI-powered assessment with detailed feedback
- ✅ **Semantic Content Retrieval**: RAG system for relevant educational content
- ✅ **Unlimited Learning**: No question limits, encouraging exploration
- ✅ **Activity-Specific Context**: Isolated sessions with comprehensive context awareness
- ✅ **Robust Fallbacks**: System continues working even when external services fail
- ✅ **Production Ready**: Full monitoring, testing, and deployment procedures

### **System Highlights**:
- **Educational Focus**: AI refuses to provide direct answers, guides learning instead
- **Context Awareness**: Understands user code, instructions, and previous feedback
- **Scalable Architecture**: Vector search, caching, and optimized database queries
- **Comprehensive Testing**: Automated test suites and manual testing procedures
- **Security First**: Input validation, content sanitization, and data protection

### **Maintenance & Support**:
- Regular monitoring of API usage and performance metrics
- Automated health checks for all system components
- Comprehensive logging for debugging and analysis
- Easy scaling through caching and database optimization
- Regular updates to AI prompts and educational content

The system is designed to evolve with the platform's needs while maintaining educational effectiveness and technical reliability.

---

**Document Version**: 1.0  
**Last Updated**: January 1, 2025  
**Maintained By**: AuraLearn Development Team

For additional support or questions, refer to the troubleshooting section or contact the development team.
