import React, { useEffect, useState } from 'react';
import { Header } from './components/Header';
import { StatusBadge } from './components/StatusBadge';
import { ControlPanel } from './components/ControlPanel';
import { ManualControl } from './components/ManualControl';
import { VideoFeed } from './components/VideoFeed';
import { useWebSocket } from './hooks/useWebSocket';

function App() {
  const { isConnected, status, lastMessage, connect, disconnect, sendCommand } = useWebSocket();
  const [isRunning, setIsRunning] = useState(false);
  const [controlMode, setControlMode] = useState('auto');

  // Handle messages from backend to update state
  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === 'response') {
        if (lastMessage.status === 'running') {
          setIsRunning(true);
        } else if (lastMessage.status === 'stopped') {
          setIsRunning(false);
        } else if (lastMessage.status === 'mode_changed') {
          setControlMode(lastMessage.mode);
        }
      }
    }
  }, [lastMessage]);

  const handleRun = () => {
    sendCommand({ type: 'command', action: 'run' });
    // Optimistic update, actual state confirmed by backend
    setIsRunning(true);
  };

  const handleStop = () => {
    sendCommand({ type: 'command', action: 'stop' });
    setIsRunning(false);
  };

  return (
    <div className="min-h-screen bg-ios-gray text-gray-900 font-sans selection:bg-ios-blue selection:text-white">
      <Header />

      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="flex flex-col gap-8">

          {/* Status Section */}
          <div className="flex items-center justify-between bg-white p-4 rounded-2xl shadow-sm border border-gray-100">
            <div className="flex flex-col">
              <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">System Status</span>
              <span className="text-lg font-medium text-gray-900">
                {isConnected ? 'System Online' : 'System Offline'}
              </span>
            </div>
            <StatusBadge status={status} />
          </div>

          {/* Video Feed */}
          <VideoFeed isRunning={isRunning} />

          {/* Controls */}
          {/* Controls */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <ControlPanel
              isConnected={isConnected}
              isRunning={isRunning}
              onConnect={connect}
              onRun={handleRun}
              onStop={handleStop}
            />
            <ManualControl
              isConnected={isConnected}
              isRunning={isRunning}
              controlMode={controlMode}
              onSendCommand={sendCommand}
            />
          </div>

          {/* Debug Info (Optional, hidden in prod usually but good for dev) */}
          <div className="mt-8 p-4 bg-gray-100 rounded-xl text-xs font-mono text-gray-500 overflow-auto max-h-32">
            <p className="font-bold mb-2">Debug Log:</p>
            <pre>{JSON.stringify(lastMessage, null, 2)}</pre>
          </div>

        </div>
      </main>
    </div>
  );
}

export default App;
