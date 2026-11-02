import { useEffect, useRef, useState } from "react";

// Quicker <-> Smarter. Cale's own control (cale/anythingllm/feel.js answers /cale-feel). No numbers and no technical words.
// The ends of the slider come from this computer, so the same position feels the same on every computer.
export default function CaleFeel() {
  const [ready, setReady] = useState(false);
  const [position, setPosition] = useState(1);
  const dragging = useRef(false);
  const timer = useRef(null);

  useEffect(() => {
    let stop = false;
    const load = () =>
      fetch("/cale-feel", { credentials: "same-origin" })
        .then((r) => r.json())
        .then((s) => {
          if (stop) return;
          setReady(!!s.ready);
          if (s.ready && !dragging.current && typeof s.position === "number") setPosition(s.position);
        })
        .catch(() => {});
    load();
    const id = setInterval(load, ready ? 20000 : 3000);
    return () => {
      stop = true;
      clearInterval(id);
    };
  }, [ready]);

  const change = (e) => {
    const p = Number(e.target.value) / 100;
    dragging.current = true;
    setPosition(p);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      fetch("/cale-feel", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ position: p }),
      })
        .catch(() => {})
        .finally(() => {
          dragging.current = false;
        });
    }, 250);
  };

  return (
    <div
      className="absolute top-3 left-1/2 -translate-x-1/2 z-20 flex items-center gap-3 rounded-full px-4 py-2 bg-zinc-800 light:bg-slate-100 text-white light:text-slate-800 text-sm select-none"
      title={ready ? "Move toward Quicker for faster answers, or toward Smarter for answers that think more." : "Cale is getting ready"}
    >
      <span className={position < 0.5 ? "font-semibold" : "opacity-70"}>Quicker</span>
      <input
        type="range"
        min="0"
        max="100"
        step="1"
        value={Math.round(position * 100)}
        onChange={change}
        disabled={!ready}
        aria-label="How Cale answers: quicker or smarter"
        className="w-40 accent-[#f58a1f] disabled:opacity-40"
      />
      <span className={position >= 0.5 ? "font-semibold" : "opacity-70"}>Smarter</span>
    </div>
  );
}
