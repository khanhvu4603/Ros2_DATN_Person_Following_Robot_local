import React, { useRef } from 'react';
import { ArrowUp, ArrowDown, ArrowLeft, ArrowRight, RotateCcw, RotateCw } from 'lucide-react';

export function ManualControl({ isConnected, isRunning, controlMode, onSendCommand }) {
  // Use a ref to track if we are currently holding a button to prevent duplicate events if needed
  // but for simple "down/up" logic, direct handlers are fine.

  const isManual = controlMode === 'manual';
  const canControl = isConnected && isRunning && isManual;

  const sendMove = (direction) => {
    if (!canControl) return;
    onSendCommand({ type: 'command', action: 'move', direction });
  };

  const sendStop = () => {
    if (!canControl) return;
    onSendCommand({ type: 'command', action: 'stop_move' });
  };

  const toggleMode = () => {
    if (!isConnected || !isRunning) return;
    const newMode = isManual ? 'auto' : 'manual';
    onSendCommand({ type: 'command', action: 'set_mode', mode: newMode });
  };

  // Helper to create button props
  const getButtonProps = (direction) => ({
    onMouseDown: () => sendMove(direction),
    onMouseUp: sendStop,
    onMouseLeave: sendStop,
    onTouchStart: (e) => { e.preventDefault(); sendMove(direction); },
    onTouchEnd: (e) => { e.preventDefault(); sendStop(); },
    disabled: !canControl,
    className: `
      p-4 rounded-xl shadow-sm border border-gray-200 
      flex items-center justify-center transition-all duration-200
      ${(!canControl)
        ? 'bg-gray-100 text-gray-300 cursor-not-allowed'
        : 'bg-white text-gray-700 hover:bg-gray-50 active:bg-blue-50 active:border-blue-200 active:text-blue-600 active:scale-95'
      }
    `
  });

  return (
    <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">Manual Control</h3>
        <div className="flex items-center gap-3">
          <span className={`text-sm font-medium ${isManual ? 'text-blue-600' : 'text-gray-400'}`}>
            {isManual ? 'MANUAL' : 'AUTO'}
          </span>
          <button
            onClick={toggleMode}
            disabled={!isConnected || !isRunning}
            className={`
                    relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
                    ${isManual ? 'bg-blue-600' : 'bg-gray-200'}
                    ${(!isConnected || !isRunning) ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                `}
          >
            <span
              className={`
                        inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                        ${isManual ? 'translate-x-6' : 'translate-x-1'}
                    `}
            />
          </button>
        </div>
      </div>

      <div className={`flex flex-col items-center gap-4 transition-opacity duration-200 ${isManual ? 'opacity-100' : 'opacity-50 pointer-events-none'}`}>
        {/* Forward */}
        <button {...getButtonProps('forward')} aria-label="Move Forward">
          <ArrowUp size={24} />
        </button>

        <div className="flex items-center gap-4">
          {/* Left */}
          <button {...getButtonProps('left')} aria-label="Move Left">
            <ArrowLeft size={24} />
          </button>

          {/* Backward (Center usually empty or maybe a stop button, but we just need layout) */}
          <div className="w-14 h-14 flex items-center justify-center text-gray-300">
            <div className="w-2 h-2 bg-current rounded-full" />
          </div>

          {/* Right */}
          <button {...getButtonProps('right')} aria-label="Move Right">
            <ArrowRight size={24} />
          </button>
        </div>

        {/* Backward */}
        <button {...getButtonProps('backward')} aria-label="Move Backward">
          <ArrowDown size={24} />
        </button>

        <div className="w-full h-px bg-gray-100 my-2" />

        {/* Rotation */}
        <div className="flex items-center gap-8">
          <button {...getButtonProps('rotate_left')} aria-label="Rotate Left">
            <RotateCcw size={24} />
          </button>
          <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">Rotate</span>
          <button {...getButtonProps('rotate_right')} aria-label="Rotate Right">
            <RotateCw size={24} />
          </button>
        </div>
      </div>
    </div>
  );
}
