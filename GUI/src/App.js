import React, { useState } from 'react';
import './App.css';

// Use environment variable for API URL, fallback to empty string for production
const API_URL = process.env.REACT_APP_API_URL || '';

function App() {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [activeModule, setActiveModule] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const modules = [
    {
      id: 'tokenization_rule',
      name: 'Tokenization (Rule-based)',
      icon: '🔤',
      gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    },
    {
      id: 'tokenization_ml',
      name: 'Tokenization (ML-based)',
      icon: '🤖',
      gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    },
    {
      id: 'sentence_rule',
      name: 'Sentence Splitting (Rule Based)',
      icon: '✂️',
      gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'
    },
    {
      id: 'sentence_nb',
      name: 'Sentence Splitting (Naive Bayes)',
      icon: '🧠',
      gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'
    },
    {
      id: 'normalization',
      name: 'Normalization',
      icon: '⚡',
      gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
    },
    {
      id: 'stemming',
      name: 'Stemming',
      icon: '🌱',
      gradient: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)'
    },
    {
      id: 'stopword_static',
      name: 'Stopword Elimination (Static)',
      icon: '🚫',
      gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)'
    },
    {
      id: 'stopword_tfidf',
      name: 'Stopword Elimination (TF-IDF)',
      icon: '📊',
      gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)'
    }
  ];

  const processText = async (moduleId) => {
    setActiveModule(moduleId);
    setIsLoading(true);
    setOutputText(''); // Clear previous output

    if (moduleId === 'tokenization_rule') {
      try {
        const response = await fetch(`${API_URL}/api/tokenize_rule`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: inputText }),
        });

        if (!response.ok) {
          throw new Error('Failed to tokenize text');
        }

        const data = await response.json();
        // Display each token on a new line
        setOutputText(data.tokens.join('\n'));
      } catch (error) {
        console.error('Error:', error);
        setOutputText('Error: Could not connect to rule-based tokenizer service');
      } finally {
        setIsLoading(false);
      }
    } else if (moduleId === 'tokenization_ml') {
      try {
        const response = await fetch(`${API_URL}/api/tokenize_ml`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: inputText }),
        });

        if (!response.ok) {
          throw new Error('Failed to tokenize text');
        }

        const data = await response.json();
        // Display each token on a new line
        setOutputText(data.tokens.join('\n'));
      } catch (error) {
        console.error('Error:', error);
        setOutputText('Error: Could not connect to ML-based tokenizer service');
      } finally {
        setIsLoading(false);
      }
    } else if (moduleId === 'stemming') {
      try {
        const response = await fetch(`${API_URL}/api/stem`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: inputText }),
        });

        if (!response.ok) {
          throw new Error('Failed to stem text');
        }

        const data = await response.json();
        // Display each word with its stem on a new line
        setOutputText(data.results.join('\n'));
      } catch (error) {
        console.error('Error:', error);
        setOutputText('Error: Could not connect to stemmer service');
      } finally {
        setIsLoading(false);
      }
    } else if (moduleId === 'stopword_static') {
      try {
        const response = await fetch(`${API_URL}/api/remove_stopwords_static`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: inputText }),
        });

        if (!response.ok) {
          throw new Error('Failed to remove stopwords');
        }

        const data = await response.json();
        // Display filtered tokens as text (with spaces between tokens)
        setOutputText(data.filtered_tokens.join(' '));
      } catch (error) {
        console.error('Error:', error);
        setOutputText('Error: Could not connect to static stopword removal service');
      } finally {
        setIsLoading(false);
      }
    } else if (moduleId === 'stopword_tfidf') {
      try {
        const response = await fetch(`${API_URL}/api/remove_stopwords_tfidf`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: inputText }),
        });

        if (!response.ok) {
          throw new Error('Failed to remove stopwords');
        }

        const data = await response.json();
        // Display filtered tokens as text (with spaces between tokens)
        setOutputText(data.filtered_tokens.join(' '));
      } catch (error) {
        console.error('Error:', error);
        setOutputText('Error: Could not connect to TF-IDF stopword removal service');
      } finally {
        setIsLoading(false);
      }
    } else if (moduleId === 'normalization') {
      try {
        const response = await fetch(`${API_URL}/api/normalize`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: inputText }),
        });

        if (!response.ok) {
          throw new Error('Failed to normalize text');
        }

        const data = await response.json();
        // Display each token with its normalized form on a new line
        setOutputText(data.results.join('\n'));
      } catch (error) {
        console.error('Error:', error);
        setOutputText('Error: Could not connect to normalization service');
      } finally {
        setIsLoading(false);
      }
    } else if (moduleId === 'sentence_rule') {
      try {
        const response = await fetch(`${API_URL}/api/split_sentences`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: inputText }),
        });

        if (!response.ok) {
          throw new Error('Failed to split sentences');
        }

        const data = await response.json();
        // Display each sentence on a new line with numbering
        const numberedSentences = data.sentences.map((sentence, index) =>
          `${index + 1}. ${sentence}`
        );
        setOutputText(numberedSentences.join('\n\n'));
      } catch (error) {
        console.error('Error:', error);
        setOutputText('Error: Could not connect to rule-based sentence splitter service');
      } finally {
        setIsLoading(false);
      }
    } else if (moduleId === 'sentence_nb') {
      try {
        const response = await fetch(`${API_URL}/api/split_sentences_nb`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: inputText }),
        });

        if (!response.ok) {
          throw new Error('Failed to split sentences');
        }

        const data = await response.json();
        // Display each sentence on a new line with numbering
        const numberedSentences = data.sentences.map((sentence, index) =>
          `${index + 1}. ${sentence}`
        );
        setOutputText(numberedSentences.join('\n\n'));
      } catch (error) {
        console.error('Error:', error);
        setOutputText('Error: Could not connect to Naive Bayes sentence splitter service');
      } finally {
        setIsLoading(false);
      }
    } else {
      // Dummy functionality for other modules - just copy input to output
      setOutputText(inputText);
      setIsLoading(false);
    }

    // Add animation effect
    setTimeout(() => {
      setActiveModule(null);
    }, 600);
  };

  return (
    <div className="App">
      <div className="background-animation"></div>

      <div className="container">
        <header className="header">
          <h1 className="title">
            <span className="title-icon">🤖</span>
            Turkish Preprocessing Toolkit
          </h1>
        </header>

        <div className="main-panel">
          <div className="text-areas">
            <div className="text-area-container">
              <label className="text-label">
                <span className="label-icon">📝</span>
                Input Text
              </label>
              <textarea
                className="text-input"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Enter your text here..."
              />
            </div>

            <div className="divider">
              <div className="divider-icon">→</div>
            </div>

            <div className="text-area-container">
              <label className="text-label">
                <span className="label-icon">✨</span>
                Output
              </label>
              <div className="output-wrapper">
                <textarea
                  className="text-output"
                  value={outputText}
                  readOnly
                  placeholder="Processed text will appear here..."
                />
                {isLoading && (
                  <div className="loading-overlay">
                    <div className="spinner"></div>
                    <p className="loading-text">Processing...</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="modules-section">
            <h2 className="modules-title">Select Processing Module</h2>
            <div className="module-buttons">
              {modules.map((module) => (
                <button
                  key={module.id}
                  className={`module-button ${activeModule === module.id ? 'active' : ''}`}
                  onClick={() => processText(module.id)}
                  style={{ '--gradient': module.gradient }}
                >
                  <span className="button-icon">{module.icon}</span>
                  <span className="button-text">{module.name}</span>
                  <span className="button-arrow">→</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        <footer className="footer">
          <p className="contact-info">
            Contact: {'{atahan.uz, gizem.yilmaz1}@std.bogazici.edu.tr'}
          </p>
        </footer>

      </div>
    </div>
  );
}

export default App;
