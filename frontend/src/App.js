import React, { useState, useRef } from "react";
import { FiUpload, FiCopy } from "react-icons/fi";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

function App() {
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [editedSummary, setEditedSummary] = useState("");
  const [summaryType, setSummaryType] = useState("quick");
  const [length, setLength] = useState("medium");
  const [isBold, setIsBold] = useState(false);
  const [isItalic, setIsItalic] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerated, setIsGenerated] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type === "text/plain") {
      const reader = new FileReader();
      reader.onload = (event) => {
        setInputText(event.target.result);
      };
      reader.readAsText(file);
    } else {
      toast.error("Please upload a valid .txt file");
    }
  };

  const handleGenerate = async () => {
    if (!inputText.trim()) {
      toast.error("Please enter some text to summarize");
      return;
    }
  
    setIsLoading(true);
    setIsGenerated(false);
    setOutputText("");  // Clear the output text
    setEditedSummary("");  // Clear the edited summary
  
    try {
      const response = await fetch("http://127.0.0.1:8000/summarize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: inputText,
          summary_type: summaryType,
          length: length,
        }),
      });
  
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
  
      const data = await response.json();
      setOutputText(data.summary);
      setEditedSummary(data.summary);
      setIsGenerated(true);
    } catch (error) {
      console.error("Error:", error);
      toast.error(`Error: ${error.message || "Failed to generate summary"}`);
      setIsGenerated(false);
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(editedSummary);
    toast.success("Summary copied to clipboard!");
  };

  return (
    <div className="min-h-screen bg-dark-1">
      <ToastContainer
        position="top-center"
        autoClose={3000}
        theme="dark"
      />
      
      <header className="border-b border-gray-700 p-4">
        <h1 className="text-2xl font-bold text-center">Text Summarizer</h1>
      </header>

      <main className="container mx-auto p-4 md:p-6 max-w-6xl">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Panel - Input */}
          <div className="bg-dark-2 rounded-xl p-6 shadow-lg">
            <h2 className="text-xl font-semibold mb-4">Input Text</h2>
            <textarea
              value={inputText}
              onChange={(e) => {
                setInputText(e.target.value);
                setIsGenerated(false);  // Reset when input changes
              }}
              className="w-full h-64 p-4 rounded-lg border border-gray-600 bg-dark-2 text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none"
              placeholder="Paste your text here or upload a file..."
            />

            <div className="mt-4 flex justify-center">
              <button
                onClick={() => fileInputRef.current.click()}
                className="flex items-center gap-2 px-4 py-2 bg-dark-3 hover:bg-dark-1 rounded-lg transition-colors"
              >
                <FiUpload /> Upload .txt File
              </button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept=".txt"
                className="hidden"
              />
            </div>

            <div className="mt-6 space-y-4">
              <div className="flex justify-between items-center">
                <label className="text-sm font-medium">Summary Type</label>
                <div className="flex gap-4">
                  <label className="flex items-center gap-1">
                    <input
                      type="radio"
                      name="summaryType"
                      checked={summaryType === "quick"}
                      onChange={() => setSummaryType("quick")}
                      className="mr-1"
                    />
                    Quick Summary
                    <div className="group relative">
                      <span className="text-gray-400 cursor-help" title="Key points extracted using a ranking method (extractive).">ⓘ</span>
                    </div>
                  </label>
                  <label className="flex items-center gap-1">
                    <input
                      type="radio"
                      name="summaryType"
                      checked={summaryType === "smart"}
                      onChange={() => setSummaryType("smart")}
                      className="mr-1"
                    />
                    Smart Summary
                    <div className="group relative">
                      <span className="text-gray-400 cursor-help" title="Concise and meaningful summary using an LLM (abstractive).">ⓘ</span>
                    </div>
                  </label>
                </div>
              </div>

              <div className="pt-2">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium">Summary length</span>
                  <span className="text-sm font-medium">Short</span>
                  <span className="text-sm font-medium">Medium</span>
                  <span className="text-sm font-medium">Long</span>
                </div>
                <div className="relative pt-1 ml-[40%]">
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="1"
                    value={["short", "medium", "long"].indexOf(length)}
                    onChange={(e) => setLength(["short", "medium", "long"][e.target.value])}
                    className="w-full h-2 bg-gray-500 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              </div>
            </div>

            <button
              onClick={handleGenerate}
              disabled={isLoading || !inputText.trim()}
              className={`w-full mt-6 py-3 rounded-lg font-medium transition-colors ${
                isLoading || !inputText.trim()
                  ? "bg-gray-600 cursor-not-allowed"
                  : "bg-primary hover:bg-primary-hover"
              }`}
            >
              {isLoading ? "Generating..." : "Generate Summary"}
            </button>
          </div>

          {/* Right Panel - Output */}
          <div className="bg-dark-2 rounded-xl p-6 shadow-lg flex flex-col h-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold flex items-center">
                Summary
                <span className={`ml-2 ${isGenerated ? 'text-green-500' : 'text-gray-500'}`}>
                  <svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    className="h-5 w-5" 
                    viewBox="0 0 20 20" 
                    fill="currentColor"
                  >
                    <path 
                      fillRule="evenodd" 
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" 
                      clipRule="evenodd" 
                    />
                  </svg>
                </span>
              </h2>
              {outputText && (
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      const textarea = document.querySelector("textarea");
                      const start = textarea.selectionStart;
                      const end = textarea.selectionEnd;
                      const selectedText = editedSummary.substring(start, end);
                      const newText = isBold 
                        ? editedSummary.substring(0, start) + selectedText + editedSummary.substring(end)
                        : editedSummary.substring(0, start) + "**" + selectedText + "**" + editedSummary.substring(end);
                      setEditedSummary(newText);
                      setIsBold(!isBold);
                    }}
                    className={`p-2 rounded ${isBold ? 'bg-dark-3' : 'hover:bg-dark-3'}`}
                    title="Bold"
                  >
                    <span className="font-bold">B</span>
                  </button>
                  <button
                    onClick={() => {
                      const textarea = document.querySelector("textarea");
                      const start = textarea.selectionStart;
                      const end = textarea.selectionEnd;
                      const selectedText = editedSummary.substring(start, end);
                      const newText = isItalic
                        ? editedSummary.substring(0, start) + selectedText + editedSummary.substring(end)
                        : editedSummary.substring(0, start) + "*" + selectedText + "*" + editedSummary.substring(end);
                      setEditedSummary(newText);
                      setIsItalic(!isItalic);
                    }}
                    className={`p-2 rounded ${isItalic ? 'bg-dark-3' : 'hover:bg-dark-3'}`}
                    title="Italic"
                  >
                    <span className="italic">I</span>
                  </button>
                  <button
                    onClick={copyToClipboard}
                    className="p-2 rounded hover:bg-dark-3"
                    title="Copy to clipboard"
                  >
                    <FiCopy />
                  </button>
                </div>
              )}
            </div>

            <div className="flex-1 flex flex-col">
              {isLoading ? (
                <div className="h-full flex items-center justify-center">
                  <div className="animate-pulse flex flex-col items-center">
                    <div className="h-2 w-24 bg-gray-600 rounded mb-2"></div>
                    <div>Generating...</div>
                  </div>
                </div>
              ) : (
                <textarea
                  value={editedSummary}
                  onChange={(e) => setEditedSummary(e.target.value)}
                  className={`w-full h-full p-4 rounded-lg border border-gray-600 bg-dark-2 text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none ${
                    isBold ? "font-bold" : ""
                  } ${isItalic ? "italic" : ""}`}
                  placeholder="Your generated summary will appear here..."
                  disabled={isLoading}
                />
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
