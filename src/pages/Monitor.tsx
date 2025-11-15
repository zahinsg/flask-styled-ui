import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useEffect, useState } from "react";
import { AlertCircle, CheckCircle2, Loader2, Wifi, WifiOff } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const Monitor = () => {
  const [protocolStatus, setProtocolStatus] = useState("CONNECTING");
  const [currentPhase, setCurrentPhase] = useState("Waiting for connection...");
  const [phaseCount, setPhaseCount] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const [backendUrl] = useState("http://localhost:5000");
  const { toast } = useToast();

  useEffect(() => {
    // Poll backend for status updates
    const pollStatus = async () => {
      try {
        const response = await fetch(`${backendUrl}/status_update`);
        if (!response.ok) throw new Error('Backend not responding');
        
        const data = await response.json();
        
        if (!isConnected) {
          setIsConnected(true);
          toast({
            title: "Connected to Flask Backend",
            description: "Live monitoring active",
          });
        }
        
        setProtocolStatus(data.result_status || "RUNNING");
        setCurrentPhase(data.current_phase || "Monitoring...");
        
        // Extract phase number from phase string
        const phaseMatch = data.current_phase?.match(/Phase (\d+)/);
        if (phaseMatch) {
          setPhaseCount(parseInt(phaseMatch[1]));
        }
      } catch (error) {
        if (isConnected) {
          setIsConnected(false);
          toast({
            title: "Connection Lost",
            description: "Make sure Flask backend is running on port 5000",
            variant: "destructive",
          });
        }
        setProtocolStatus("DISCONNECTED");
        setCurrentPhase("Backend not connected");
      }
    };

    // Initial connection
    pollStatus();
    
    // Poll every 500ms for real-time updates
    const interval = setInterval(pollStatus, 500);

    return () => clearInterval(interval);
  }, [isConnected, backendUrl, toast]);

  const getStatusColor = () => {
    if (protocolStatus.includes("PASS") || protocolStatus.includes("VERIFIED")) return "bg-success";
    if (protocolStatus.includes("FAIL") || protocolStatus.includes("FATAL")) return "bg-destructive";
    if (protocolStatus === "DISCONNECTED") return "bg-muted";
    return "bg-warning";
  };

  const getStatusIcon = () => {
    if (protocolStatus.includes("PASS") || protocolStatus.includes("VERIFIED")) return <CheckCircle2 className="w-5 h-5" />;
    if (protocolStatus.includes("FAIL") || protocolStatus.includes("FATAL")) return <AlertCircle className="w-5 h-5" />;
    if (protocolStatus === "DISCONNECTED") return <WifiOff className="w-5 h-5" />;
    return <Loader2 className="w-5 h-5 animate-spin" />;
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">💊 Medication Adherence Protocol</h1>
          <p className="text-muted-foreground">Follow the visual instructions on the screen. The process is being monitored in real-time.</p>
        </div>

        <Card className="p-6 shadow-elevated mb-6">
          <div className="relative aspect-video bg-muted rounded-lg flex items-center justify-center border-4 border-border overflow-hidden">
            {isConnected ? (
              <img 
                src={`${backendUrl}/video_feed`}
                alt="Live Video Feed"
                className="w-full h-full object-contain"
                onError={() => {
                  console.error("Video feed error");
                }}
              />
            ) : (
              <div className="text-center p-8">
                <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
                  <WifiOff className="w-12 h-12 text-muted-foreground animate-pulse" />
                </div>
                <p className="text-muted-foreground mb-2">Waiting for Flask Backend</p>
                <p className="text-xs text-muted-foreground">Run: python proto.py</p>
                <p className="text-xs text-muted-foreground mt-1">Backend URL: {backendUrl}</p>
              </div>
            )}
            
            {/* Connection Status Badge */}
            <div className="absolute top-3 right-3">
              <Badge variant={isConnected ? "default" : "secondary"} className="flex items-center gap-1">
                {isConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
                {isConnected ? "Connected" : "Offline"}
              </Badge>
            </div>
          </div>
        </Card>

        <Card className="p-6 shadow-card mb-6">
          <div className={`${getStatusColor()} text-white rounded-lg p-6 flex items-center justify-between`}>
            <div className="flex items-center space-x-3">
              {getStatusIcon()}
              <span className="font-bold text-lg">Protocol Status: {protocolStatus}</span>
            </div>
            <Badge variant="secondary" className="bg-white/20 text-white">
              Live
            </Badge>
          </div>
        </Card>

        <Card className="p-6 shadow-card">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-medium text-muted-foreground mb-1">Current Phase</h3>
              <p className="text-lg font-semibold text-foreground">{currentPhase}</p>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold text-primary">{phaseCount}/6</div>
              <div className="text-xs text-muted-foreground">Phases</div>
            </div>
          </div>
          
          <div className="mt-6">
            <div className="flex justify-between mb-2">
              {[1, 2, 3, 4, 5, 6].map((phase) => (
                <div
                  key={phase}
                  className={`w-full h-2 mx-0.5 rounded-full transition-all ${
                    phase <= phaseCount ? "bg-primary" : "bg-muted"
                  }`}
                />
              ))}
            </div>
          </div>
        </Card>

        <div className="mt-6 p-4 bg-accent rounded-lg">
          <h4 className="font-semibold text-sm text-accent-foreground mb-2">Protocol Instructions:</h4>
          <ul className="text-sm text-muted-foreground space-y-1">
            <li>• Keep your face visible in the camera frame</li>
            <li>• Follow each phase as instructed</li>
            <li>• Allow time for detection and verification</li>
            <li>• The process completes automatically upon success</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

// Import Activity icon from lucide-react
import { Activity } from "lucide-react";

export default Monitor;
