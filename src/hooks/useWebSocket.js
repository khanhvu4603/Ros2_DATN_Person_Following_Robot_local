import { useState, useEffect, useRef, useCallback } from 'react';

const WS_URL = 'ws://100.114.119.34:8000/ws';

export const useWebSocket = () => {
    const [isConnected, setIsConnected] = useState(false);
    const [status, setStatus] = useState('disconnected'); // disconnected, connecting, connected
    const [lastMessage, setLastMessage] = useState(null);
    const wsRef = useRef(null);

    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        setStatus('connecting');
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
            setIsConnected(true);
            setStatus('connected');
            console.log('WebSocket Connected');
            // Send initial connect message
            ws.send(JSON.stringify({ type: 'connect', timestamp: Date.now() }));
        };

        ws.onclose = () => {
            setIsConnected(false);
            setStatus('disconnected');
            console.log('WebSocket Disconnected');
            wsRef.current = null;
        };

        ws.onerror = (error) => {
            console.error('WebSocket Error:', error);
            ws.close();
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                setLastMessage(data);
                console.log('Received:', data);
            } catch (e) {
                console.error('Parse error:', e);
            }
        };
    }, []);

    const disconnect = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.close();
        }
    }, []);

    const sendCommand = useCallback((command) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(command));
        } else {
            console.warn('WebSocket not connected');
        }
    }, []);

    useEffect(() => {
        // Cleanup on unmount
        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, []);

    return {
        isConnected,
        status,
        lastMessage,
        connect,
        disconnect,
        sendCommand
    };
};
