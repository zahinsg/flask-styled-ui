import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useEffect, useState } from "react";
import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react";

const Monitor = () => {
  const [protocolStatus, setProtocolStatus] = useState("RUNNING");
  const [currentPhase, setCurrentPhase] = useState("Phase 1: Initial Detection");
  const [phaseCount, setPhaseCount] = useState(1);

  useEffect(() => {
    // Simulate protocol progression
    const interval = setInterval(() => {
      setPhaseCount((prev) => {
        const next = prev >= 6 ? 1 : prev + 1;
        
        const phases = [
          "Phase 1: Initial Detection",
          "Phase 2: Hand Movement",
          "Phase 3: Pill on Tongue",
          "Phase 4: Concealment Check",
          "Phase 5: Mouth Closure",
          "Phase 6: Final Verification"
        ];
        
        setCurrentPhase(phases[next - 1]);
        
        if (next === 6) {
          setTimeout(() => {
            setProtocolStatus("VERIFIED (PASS)");
          }, 2000);
        } else if (next === 1) {
          setProtocolStatus("RUNNING");
        }
        
        return next;
      });
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = () => {
    if (protocolStatus === "VERIFIED (PASS)") return "bg-success";
    if (protocolStatus === "FAIL") return "bg-destructive";
    return "bg-warning";
  };

  const getStatusIcon = () => {
    if (protocolStatus === "VERIFIED (PASS)") return <CheckCircle2 className="w-5 h-5" />;
    if (protocolStatus === "FAIL") return <AlertCircle className="w-5 h-5" />;
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
          <div className="aspect-video bg-muted rounded-lg flex items-center justify-center border-4 border-border overflow-hidden">
            <div className="text-center p-8">
              <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
                <Activity className="w-12 h-12 text-primary animate-pulse" />
              </div>
              <p className="text-muted-foreground mb-2">Live Video Feed</p>
              <p className="text-xs text-muted-foreground">Connect to Flask backend at /video_feed for live stream</p>
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
