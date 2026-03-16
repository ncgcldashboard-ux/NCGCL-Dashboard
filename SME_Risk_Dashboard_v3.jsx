
import { useState, useEffect } from "react";

// ─── DUAL BRAND SYSTEM ────────────────────────────────────────────────────────
// Anthropic: warm parchment shell, editorial serif/humanist type, terracotta accents
// NCGCL:     cerulean/green as the data engine — all charts, numbers, and actions

const A = {
  // Anthropic core
  dark:        "#141413",
  light:       "#faf9f5",
  midGray:     "#b0aea5",
  lightGray:   "#e8e6dc",
  orange:      "#d97757",  // Anthropic primary accent
  blue:        "#6a9bcc",  // Anthropic secondary accent
  sage:        "#788c5d",  // Anthropic tertiary accent
  cream:       "#f3f1ea",
  warmWhite:   "#faf9f5",
  stone:       "#c9c7be",
  charcoal:    "#2c2b28",
  // NCGCL data engine
  cerulean:    "#035076",
  green:       "#39A949",
  cyan:        "#11688A",
  amber:       "#C75000",
  red:         "#b83232",
  // Semantic
  text:        "#2c2b28",
  textMid:     "#6b6960",
  textLight:   "#9e9c95",
  // Surface
  surface:     "#ffffff",
  surfaceWarm: "#faf9f5",
  border:      "#e8e6dc",
  borderWarm:  "#d8d6ce",
};

// ─── DATA ── (unchanged from v2) ─────────────────────────────────────────────

const timeSeriesData = [
  { period: "Jun-24",  outstanding: 570.1,  growth: null,   mom: null,   phase: 1, note: "RCS launch baseline" },
  { period: "Jun-25",  outstanding: 756.3,  growth: 32.7,   mom: null,   phase: 2, note: "FY25 close" },
  { period: "Sep-25",  outstanding: 769.4,  growth: 1.7,    mom: 1.7,    phase: 2, note: "" },
  { period: "Oct-25",  outstanding: 810.4,  growth: 5.3,    mom: 5.3,    phase: 3, note: "Acceleration" },
  { period: "Nov-25",  outstanding: 857.6,  growth: 5.8,    mom: 5.8,    phase: 3, note: "" },
  { period: "Dec-25",  outstanding: 947.9,  growth: 10.5,   mom: 10.5,   phase: 3, note: "⚠ Verify: Large year-end jump", alert: true },
  { period: "Jan-26",  outstanding: 937.0,  growth: -1.1,   mom: -1.1,   phase: 3, note: "Latest data (SBP Jan-26)" },
];

const q4momData = [
  { period: "Sep→Oct-25", delta: 41.0, mom: 5.3, top: [["Wholesale/Retail (G)", 43], ["Manufacturing (C)", 29], ["Other Services (Q)", 11]] },
  { period: "Oct→Nov-25", delta: 47.1, mom: 5.8, top: [["Manufacturing (C)", 42], ["Wholesale/Retail (G)", 37], ["Transport (H)", 6]] },
  { period: "Nov→Dec-25", delta: 90.4, mom: 10.5, top: [["Manufacturing (C)", 33], ["Wholesale/Retail (G)", 31], ["Agriculture (A)", 22]], alert: true },
];

const sectorData = [
  { sector: "Wholesale & Retail Trade",   code: "G", dec25: 325.8, share: 34.4, growth: 63.6, contrib: 33.5, risk: "High",       color: A.cerulean, tfin: 5212.6,   wc: 178010.3, fi: 101078.2 },
  { sector: "Manufacturing",              code: "C", dec25: 316.9, share: 33.4, growth: 56.0, contrib: 30.1, risk: "Moderate",   color: A.blue,     tfin: 29442.5,  wc: 215135.7, fi: 58047.5  },
  { sector: "Agriculture & Fishing",      code: "A", dec25: 95.3,  share: 10.1, growth: 91.7, contrib: 12.1, risk: "High",       color: A.sage,     tfin: 192.3,    wc: 38725.6,  fi: 34566.5  },
  { sector: "Other Service Activities",   code: "Q", dec25: 60.2,  share: 6.4,  growth: 153.2,contrib: 9.6,  risk: "Very High",  color: A.red,      tfin: 192.5,    wc: 32599.5,  fi: 23178.0  },
  { sector: "Transportation & Storage",   code: "H", dec25: 56.2,  share: 5.9,  growth: 97.2, contrib: 7.3,  risk: "Moderate",   color: A.cyan,     tfin: 0,        wc: 2519.2,   fi: 32332.0  },
  { sector: "Construction",               code: "F", dec25: 21.8,  share: 2.3,  growth: 41.3, contrib: 1.7,  risk: "Moderate",   color: A.midGray,  tfin: 59.4,     wc: 7309.9,   fi: 7318.8   },
  { sector: "Professional & Scientific",  code: "L", dec25: 15.9,  share: 1.7,  growth: -13.9,contrib: -0.7, risk: "Moderate",   color: A.stone,    tfin: 245.5,    wc: 8962.7,   fi: 6508.6   },
  { sector: "Admin & Support Services",   code: "M", dec25: 12.7,  share: 1.3,  growth: 55.6, contrib: 1.2,  risk: "Moderate",   color: "#8a7f6e",  tfin: 376.1,    wc: 6194.1,   fi: 5717.1   },
  { sector: "ICT",                        code: "J", dec25: 11.2,  share: 1.2,  growth: 128.9,contrib: 1.7,  risk: "High",       color: A.orange,   tfin: 70.0,     wc: 5242.0,   fi: 4827.1   },
  { sector: "Accommodation & Food",       code: "I", dec25: 10.7,  share: 1.1,  growth: 106.3,contrib: 1.5,  risk: "High",       color: "#b87040",  tfin: 1226.4,   wc: 2706.0,   fi: 5033.9   },
];

const borrowerData = [
  { period: "Sep-24", outstanding: 478.37, borrowers: 177753, avgTicket: 2.69 },
  { period: "Jun-25", outstanding: 690.98, borrowers: 276593, avgTicket: 2.50 },
  { period: "Sep-25", outstanding: 686.17, borrowers: 295291, avgTicket: 2.32 },
];

const facilityData = [
  { type: "Fixed Investment", sep24: 207.22, sep25: 340.28, shareSep24: 43.3, shareSep25: 49.6, change: 6.3 },
  { type: "Working Capital",  sep24: 248.21, sep25: 317.55, shareSep24: 51.9, shareSep25: 46.3, change: -5.6 },
  { type: "Trade Finance",    sep24: 22.94,  sep25: 28.34,  shareSep24: 4.8,  shareSep25: 4.1,  change: -0.7 },
];

const bankingData = [
  { channel: "Domestic Private Banks",         sep24: 287.55, sep25: 369.96, growth: 28.7, shareSep25: 53.9, trend: "stable",   color: A.cerulean },
  { channel: "Public Sector Comm. Banks",      sep24: 128.51, sep25: 220.18, growth: 71.3, shareSep25: 32.1, trend: "rising",   color: A.amber },
  { channel: "Islamic Banks",                  sep24: 48.90,  sep25: 82.98,  growth: 69.7, shareSep25: 12.1, trend: "rising",   color: A.green },
  { channel: "Specialized Banks & Others",     sep24: 7.48,   sep25: 5.50,   growth: -26.5,shareSep25: 0.8,  trend: "declining",color: A.midGray },
  { channel: "DFIs",                           sep24: 5.94,   sep25: 7.55,   growth: 27.1, shareSep25: 1.1,  trend: "stable",   color: A.blue },
];

const smeTypeData = [
  { type: "Trading SMEs",       sep24: 208.20, sep25: 284.31, growth: 36.6, shareSep25: 41.4, color: A.cerulean },
  { type: "Services SMEs",      sep24: 129.78, sep25: 222.40, growth: 71.4, shareSep25: 32.4, color: A.orange },
  { type: "Manufacturing SMEs", sep24: 140.39, sep25: 179.46, growth: 27.8, shareSep25: 26.2, color: A.green },
];

const segmentData = [
  { segment: "Corporate",   adv24: 12304672, adv25: 10896483, npl24: 755753,  npl25: 672208,  ir24: 6.1,  ir25: 6.2,  advGrowth: -11.4, nplChange: -11.0, color: A.midGray },
  { segment: "SMEs",        adv24: 677718,   adv25: 902935,   npl24: 122210,  npl25: 113049,  ir24: 18.0, ir25: 12.5, advGrowth: 33.2,  nplChange: -7.5,  color: A.cerulean },
  { segment: "Agriculture", adv24: 578498,   adv25: 673606,   npl24: 56852,   npl25: 102777,  ir24: 9.8,  ir25: 15.3, advGrowth: 16.4,  nplChange: 80.8,  color: A.red },
  { segment: "Consumer",    adv24: 891241,   adv25: 1031707,  npl24: 38465,   npl25: 44409,   ir24: 4.3,  ir25: 4.3,  advGrowth: 15.8,  nplChange: 15.5,  color: A.green },
];

const elScenarios = [
  { scenario: "Optimistic", pd: 7.2,  lgd: 53.0, el: 36.2, ead: 947.9, color: A.green,    note: "Improving asset quality persists; benign macro" },
  { scenario: "Base Case",  pd: 9.0,  lgd: 63.0, el: 53.7, ead: 947.9, color: A.blue,     note: "Conservative base given unseasoned growth" },
  { scenario: "Stress",     pd: 10.8, lgd: 73.0, el: 74.7, ead: 947.9, color: A.red,      note: "Macro shock; weaker recoveries on new tickets" },
];

const fiscalData = [
  { scenario: "Benign",  proxyEAD: 377.8, firstLoss: 54.8, pd: 6.0,  claims: 3.3, color: A.green },
  { scenario: "Central", proxyEAD: 377.8, firstLoss: 54.8, pd: 8.5,  claims: 4.7, color: A.blue  },
  { scenario: "Stress",  proxyEAD: 377.8, firstLoss: 54.8, pd: 13.5, claims: 7.4, color: A.red   },
];

const annexAData = [
  { sector: "G. Wholesale & Retail Trade",         total: 325789.5, tfin: 5212.6,  wc: 178010.3, fi: 101078.2, const_: 198.5,   other: 41290.0  },
  { sector: "C. Manufacturing",                    total: 316855.3, tfin: 29442.5, wc: 215135.7, fi: 58047.5,  const_: 59.7,    other: 14169.9  },
  { sector: "A. Agriculture, Forestry & Fishing",  total: 95301.9,  tfin: 192.3,   wc: 38725.6,  fi: 34566.5,  const_: 9.1,     other: 21808.4  },
  { sector: "Q. Other Service Activities",         total: 60242.8,  tfin: 192.5,   wc: 32599.5,  fi: 23178.0,  const_: 25.3,    other: 4247.5   },
  { sector: "H. Transportation & Storage",         total: 56186.5,  tfin: 0,       wc: 2519.2,   fi: 32332.0,  const_: 92.5,    other: 21242.7  },
  { sector: "F. Construction",                     total: 21827.2,  tfin: 59.4,    wc: 7309.9,   fi: 7318.8,   const_: 6568.8,  other: 570.3    },
  { sector: "L. Professional & Scientific",        total: 15859.9,  tfin: 245.5,   wc: 8962.7,   fi: 6508.6,   const_: 0,       other: 143.1    },
  { sector: "M. Admin & Support Services",         total: 12685.3,  tfin: 376.1,   wc: 6194.1,   fi: 5717.1,   const_: 0,       other: 398.0    },
  { sector: "J. ICT",                             total: 11167.7,  tfin: 70.0,    wc: 5242.0,   fi: 4827.1,   const_: 101.5,   other: 927.0    },
  { sector: "I. Accommodation & Food",             total: 10714.0,  tfin: 1226.4,  wc: 2706.0,   fi: 5033.9,   const_: 1009.0,  other: 738.7    },
  { sector: "N. Education",                        total: 5905.5,   tfin: 0,       wc: 1790.8,   fi: 3460.2,   const_: 47.8,    other: 606.8    },
  { sector: "O. Human Health & Social Work",       total: 4504.9,   tfin: 0,       wc: 1382.9,   fi: 2701.1,   const_: 33.3,    other: 387.5    },
  { sector: "D. Electricity, Gas & Steam",         total: 3488.8,   tfin: 9.3,     wc: 1876.2,   fi: 1590.2,   const_: 0,       other: 13.2     },
  { sector: "K. Real Estate",                      total: 3408.5,   tfin: 0,       wc: 681.3,    fi: 2459.7,   const_: 243.8,   other: 23.7     },
  { sector: "B. Mining & Quarrying",               total: 2200.2,   tfin: 5.0,     wc: 1003.2,   fi: 1178.5,   const_: 0,       other: 13.5     },
  { sector: "P. Arts, Entertainment & Recreation", total: 928.9,    tfin: 100.0,   wc: 340.0,    fi: 477.3,    const_: 0.9,     other: 10.8     },
  { sector: "E. Water Supply & Waste Management",  total: 861.9,    tfin: 0,       wc: 72.0,     fi: 707.4,    const_: 0,       other: 82.5     },
];

const audienceMessages = {
  ceo: {
    headline: "Your SME book grew +66.3% in 18 months. The Dec-25 surge (+10.5% MoM) needs urgent validation.",
    bullets: [
      "Dec-25 year-end spike: +PKR 90.4 Bn in a single month. 85% in 3 sectors. Verify: seasonal WC drawdown or genuine acceleration?",
      "Fixed Investment now 49.6% of SME book (up from 43.3%). Long-dated, illiquid exposure — refinancing risk at sustained high rates.",
      "Services SMEs grew +71.4% — highest among SME types. Public sector banks led (+71.3%) — weakest underwriting channel.",
      "Agriculture NPLs rose +80.8% even as total system NPLs fell. Infection ratio: 9.8% → 15.3%. Now highest among all segments.",
      "Average ticket COMPRESSED -13.6% (PKR 2.69M → 2.32M). More borrowers, smaller tickets = higher admin cost + default risk per PKR.",
    ]
  },
  policy: {
    headline: "Three SBP tables give three different SME numbers. A reconciliation map is required before any budget figure is final.",
    bullets: [
      "First-loss exposure under RCS: PKR 54.8 Bn on proxy EAD of PKR 377.8 Bn (net growth Jun-24 to Dec-25).",
      "Expected claims (12-month horizon): PKR 4.7 Bn central | PKR 7.4 Bn stress. Budget accordingly.",
      "Corporate credit contracted PKR 1.4 Trillion (-11.4%) while SME grew +33.2%. Credit reallocation = macro significance.",
      "Agriculture sector infection ratio: 9.8% → 15.3% in one year. Agriculture NPLs up +80.8%. Growing fiscal risk.",
      "Dec-25 +10.5% MoM jump in SME outstanding must be validated — seasonal drawdown or reporting artifact would change fiscal liability estimate.",
    ]
  },
  regulator: {
    headline: "Services SMEs: +71.4% growth. Public sector banks: +71.3% growth. Both require targeted supervisory action.",
    bullets: [
      "Definition drift: Table 3.14 (ISIC) ≠ Table 1.9 (segment) ≠ Sep SBP SME. Three datasets, three different exposure numbers. Reconcile immediately.",
      "Dec-25 +10.5% MoM spike: verify whether seasonal WC drawdowns, reporting lags, reclassifications, or genuine credit acceleration.",
      "Borrower expansion (177K → 295K) not matched by proportional outstanding growth — denominator diluting NPL ratios. Vintage analysis mandatory.",
      "Public sector banks' SME share: 26.9% → 32.1%. Historically weakest underwriting. Program-driven targets suspected.",
      "Professional/Scientific sector DECLINED -13.9% — unique anomaly. Could signal de-risking or SBP definitional adjustment.",
    ]
  }
};

// ─── DESIGN COMPONENTS ───────────────────────────────────────────────────────

// Logo: actual NCGCL SVG — white/green/blue paths render perfectly on dark header
const NCGCLLogo = ({ height = 36 }) => (
  <svg height={height} viewBox="0 0 222.55 46.57" xmlns="http://www.w3.org/2000/svg" style={{ display: "block" }}>
    <path fill="#156a8b" d="M0,0l1.05.21,4.84,3.36,3.37,2.31,2.1,1.47,3.37,2.31,4.84,3.36,2.95,2.1,1.68,1.68,2.1,3.36,1.68,2.52,1.26-.21,5.26-1.89,1.89.21,4.63,1.68h1.26l1.05-1.05,1.68-2.94,1.47-2.1,1.05-1.05,2.1-1.68,3.16-2.1,2.31-1.68,4.63-3.15,3.37-2.31,3.37-2.31,2.74-1.89,1.05-.21v1.68l-.84,4.2-1.05,1.05-3.37,2.52-2.53,1.89-.42.42,4.63-2.52,1.05-.63,1.05.21-.21,1.89-.63,3.78-.84,1.26-4.63,2.94-3.79,2.31,4.21-1.68,1.05-.42,1.05.21-.21,1.05-1.89,3.15-3.16,5.45-2.1,3.57-1.26,1.05-2.95,1.89-2.31,1.47-4.63,2.94-4.84,2.94-4,2.52-2.1,1.26-1.26.21-1.68-.84-2.1-1.47-7.37-4.82-6.1-3.99-3.16-2.1-1.26-1.05-3.16-5.45-2.1-3.57-2.1-3.57.21-.84,1.47.21,4.42,1.89-1.89-1.26-4.42-2.73-2.1-1.47-.63-1.05-.84-4.82.21-1.05,1.26.21,5.26,2.94-1.47-1.26-4-2.94-1.89-1.68L0,1.89V0ZM61.24,11.96l-.21.42.63-.21-.42-.21Z"/>
    <path fill="#3ba949" d="M69.67,0h.62v1.69l-.83,4.22-1.04,1.06-3.32,2.53-2.49,1.9-.41.42,4.56-2.53,1.04-.63,1.04.21-.21,1.9-.62,3.8-.83,1.27-4.56,2.95-3.73,2.32,4.15-1.69,1.04-.42,1.04.21-.21,1.06-1.87,3.17-3.11,5.49-2.07,3.59-1.24,1.06-2.9,1.9-2.28,1.48-1.24.84-1.04-.42-2.9-2.11-3.73-2.74-3.94-2.74-3.32-2.53-1.24-.21-3.53,1.9h-1.24l-2.07-1.9-1.66-1.9.21-.84,2.7-1.06,2.49-1.06,3.86-1.9,2.23.4,4.09,1.5h1.43l.43-1.32,2.1-2.53,1.62-2.31,1.49-1.12,1.49-1.49,3.35-2.23,2.42-1.67,5.02-3.36,2.98-2.05,3.16-1.98,2.7-1.9.41-.21ZM61.37,12.03l-.21.42.62-.21-.41-.21Z"/>
    <circle fill="#fff" cx="28.9" cy="25.42" r=".8"/>
    <path fill="#fff" d="M88.05,38.35V9.94h2.84c5.61,8.01,11.21,15.95,16.86,24.29h.07V9.94h2.62v28.4h-2.84c-5.53-7.9-11.21-15.73-16.89-23.96h-.04v23.96h-2.62ZM129.1,38.96c-8.41,0-14.35-6.04-14.35-14.71s6.34-14.89,14.16-14.89c5.28,0,10.67,2.69,12.53,8.3l-2.37.95c-1.71-4.33-5.68-6.81-10.12-6.81-6.15,0-11.43,4.66-11.43,12.49,0,8.3,5.94,12.23,11.83,12.23s9.54-3.93,10.85-8.59l2.33.84c-1.78,5.83-6.26,10.2-13.44,10.2ZM157.93,38.96c-8.19,0-14.06-6.08-14.06-14.71s6.26-14.89,14.16-14.89c5.32,0,10.67,2.73,12.53,8.3l-2.37.95c-1.71-4.33-5.68-6.81-10.12-6.81-6.12,0-11.43,4.62-11.43,12.49,0,8.34,5.97,12.23,11.72,12.23s10.05-4.19,10.49-9.79h-9.87v-2.26h12.45v13.87h-2.15c0-2.11-.04-4.3-.07-6.41h-.18c-1.93,4.37-5.64,7.03-11.11,7.03ZM188.73,38.96c-8.41,0-14.35-6.04-14.35-14.71s6.33-14.89,14.16-14.89c5.28,0,10.67,2.69,12.53,8.3l-2.37.95c-1.71-4.33-5.68-6.81-10.12-6.81-6.15,0-11.43,4.66-11.43,12.49,0,8.3,5.93,12.23,11.83,12.23s9.54-3.93,10.85-8.59l2.33.84c-1.78,5.83-6.26,10.2-13.44,10.2ZM205.4,38.35V9.94h2.62v25.93h14.53v2.48h-17.15Z"/>
  </svg>
);

// Anthropic-flavored slash mark used as divider / decoration
const Slash = ({ color = A.orange, size = 14 }) => (
  <span style={{ fontFamily: "'Poppins',sans-serif", fontWeight: 700, color, fontSize: size, margin: "0 6px", opacity: 0.7 }}>/</span>
);

// Section header with Poppins title + Lora subtitle — Anthropic editorial feel
const SectionHeader = ({ number, title, subtitle }) => (
  <div style={{ marginBottom: 28, paddingBottom: 16, borderBottom: `1.5px solid ${A.border}` }}>
    <div style={{ display: "flex", alignItems: "flex-start", gap: 14 }}>
      <div style={{
        width: 40, height: 40, background: A.cerulean, borderRadius: 6,
        display: "flex", alignItems: "center", justifyContent: "center",
        color: A.light, fontWeight: 700, fontSize: 15,
        fontFamily: "'Poppins',sans-serif", flexShrink: 0, marginTop: 2
      }}>{number}</div>
      <div>
        <div style={{ fontSize: 20, fontWeight: 700, color: A.dark, fontFamily: "'Poppins',sans-serif", lineHeight: 1.2 }}>{title}</div>
        {subtitle && <div style={{ fontSize: 12.5, color: A.textMid, marginTop: 4, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>{subtitle}</div>}
      </div>
    </div>
  </div>
);

// Metric card — warm surface, Poppins number, Lora label
const StatCard = ({ label, value, sub, color = A.cerulean, warning = false, size = "md" }) => {
  const isLg = size === "lg";
  const [hovered, setHovered] = useState(false);
  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: A.surface,
        borderRadius: 10,
        padding: isLg ? "20px 22px" : "14px 18px",
        borderLeft: `3px solid ${warning ? A.red : color}`,
        boxShadow: hovered ? "0 6px 28px rgba(20,20,19,0.10)" : "0 1px 6px rgba(20,20,19,0.06)",
        transition: "box-shadow 0.22s, transform 0.22s",
        transform: hovered ? "translateY(-2px)" : "none",
        cursor: "default",
        borderTop: `1px solid ${A.border}`,
        borderRight: `1px solid ${A.border}`,
        borderBottom: `1px solid ${A.border}`,
      }}>
      <div style={{ fontSize: 9.5, fontWeight: 600, color: A.textLight, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 6, fontFamily: "'Poppins',sans-serif" }}>{label}</div>
      <div style={{ fontSize: isLg ? 27 : 23, fontWeight: 700, color: warning ? A.red : color, fontFamily: "'Poppins',sans-serif", lineHeight: 1, letterSpacing: "-0.01em" }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: A.textMid, marginTop: 6, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic", lineHeight: 1.45 }}>{sub}</div>}
    </div>
  );
};

// Alert/callout box — Anthropic's warm tints, Lora body
const AlertBox = ({ type = "info", title, children }) => {
  const variants = {
    warning:  { bg: "#fef5ec", border: A.orange,   icon: "⚠",  ic: A.orange   },
    success:  { bg: "#f2f7ee", border: A.sage,     icon: "✓",  ic: A.sage     },
    info:     { bg: "#eef4f8", border: A.blue,     icon: "→",  ic: A.blue     },
    critical: { bg: "#fdf0f0", border: A.red,      icon: "!",  ic: A.red      },
    new:      { bg: "#f2f7ee", border: A.green,    icon: "★",  ic: A.green    },
  };
  const s = variants[type] || variants.info;
  return (
    <div style={{
      display: "flex", gap: 11,
      background: s.bg,
      borderLeft: `3px solid ${s.border}`,
      borderRadius: "0 8px 8px 0",
      padding: "10px 14px", marginBottom: 10,
    }}>
      <div style={{ fontSize: 13, color: s.ic, fontWeight: 700, flexShrink: 0, marginTop: 1 }}>{s.icon}</div>
      <div>
        {title && <div style={{ fontWeight: 600, color: s.ic, fontSize: 11.5, marginBottom: 3, fontFamily: "'Poppins',sans-serif" }}>{title}</div>}
        <div style={{ fontSize: 11.5, color: A.text, lineHeight: 1.65, fontFamily: "'Lora','Georgia',serif" }}>{children}</div>
      </div>
    </div>
  );
};

const RiskBadge = ({ level }) => {
  const map = {
    "Low":       ["#eef6f0", A.sage],
    "Moderate":  ["#eef3f7", A.blue],
    "High":      ["#fef0e8", A.orange],
    "Very High": ["#fdf0f0", A.red],
    "Anomaly ⚠": ["#fef9ec", "#9a6c00"],
  };
  const [bg, col] = map[level] || map["Moderate"];
  return (
    <span style={{
      background: bg, color: col,
      padding: "2px 9px", borderRadius: 20,
      fontSize: 10, fontWeight: 600,
      fontFamily: "'Poppins',sans-serif", whiteSpace: "nowrap",
      border: `1px solid ${col}30`,
    }}>{level}</span>
  );
};

// Horizontal bar — minimal Anthropic style
const HBar = ({ label, value, max, color, sub, negative }) => (
  <div style={{ marginBottom: 11 }}>
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
      <span style={{ fontSize: 12, fontFamily: "'Lora','Georgia',serif", color: A.text }}>{label}</span>
      <span style={{ fontSize: 12, fontWeight: 700, color: negative ? A.red : color, fontFamily: "'Poppins',sans-serif" }}>{value}</span>
    </div>
    <div style={{ height: 5, background: A.lightGray, borderRadius: 3, overflow: "hidden" }}>
      <div style={{ height: "100%", width: `${Math.abs(parseFloat(value)) / max * 100}%`, background: negative ? A.red : color, borderRadius: 3 }} />
    </div>
    {sub && <div style={{ fontSize: 10, color: A.textLight, marginTop: 2, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>{sub}</div>}
  </div>
);

// Sparkline chart
const SparkLine = ({ data, color = A.cerulean, height = 72, yKey = "outstanding" }) => {
  const vals = data.map(d => d[yKey]);
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const w = 400, h = height;
  const pts = vals.map((v, i) => {
    const x = (i / (vals.length - 1)) * (w - 20) + 10;
    const y = h - ((v - mn) / (mx - mn || 1)) * (h - 22) - 11;
    return `${x},${y}`;
  }).join(" ");
  const area = `10,${h} ${pts} ${w - 10},${h}`;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: "100%", height }} preserveAspectRatio="none">
      <defs>
        <linearGradient id="sg3" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.15" />
          <stop offset="100%" stopColor={color} stopOpacity="0.01" />
        </linearGradient>
      </defs>
      <polygon points={area} fill="url(#sg3)" />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
      {vals.map((v, i) => {
        const x = (i / (vals.length - 1)) * (w - 20) + 10;
        const y = h - ((v - mn) / (mx - mn || 1)) * (h - 22) - 11;
        const isAlert = data[i].alert;
        return <circle key={i} cx={x} cy={y} r={i === vals.length - 1 ? 5.5 : isAlert ? 5 : 2.5}
          fill={isAlert ? A.orange : i === vals.length - 1 ? color : color}
          stroke={isAlert ? A.surface : "none"} strokeWidth="1.5"
          opacity={i === vals.length - 1 || isAlert ? 1 : 0.45} />;
      })}
    </svg>
  );
};

// Semi-circular gauge — refined minimal look
const Gauge = ({ value, max, label, color, size = 90 }) => {
  const pct = Math.min(value / max, 1);
  const r = size / 2 - 11;
  const arc = pct * Math.PI * r;
  const total = Math.PI * r;
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
      <svg width={size} height={size / 2 + 12} viewBox={`0 0 ${size} ${size / 2 + 12}`}>
        <path d={`M 11 ${size / 2} A ${r} ${r} 0 0 1 ${size - 11} ${size / 2}`}
          fill="none" stroke={A.lightGray} strokeWidth="9" strokeLinecap="round" />
        <path d={`M 11 ${size / 2} A ${r} ${r} 0 0 1 ${size - 11} ${size / 2}`}
          fill="none" stroke={color} strokeWidth="9" strokeLinecap="round"
          strokeDasharray={`${arc} ${total}`} />
        <text x={size / 2} y={size / 2 - 1} textAnchor="middle"
          fill={color} fontSize="14" fontWeight="700" fontFamily="'Poppins',sans-serif">{value}%</text>
      </svg>
      <div style={{ fontSize: 10, fontWeight: 500, color: A.textMid, textAlign: "center", fontFamily: "'Poppins',sans-serif" }}>{label}</div>
    </div>
  );
};

// ─── MAIN SHELL ───────────────────────────────────────────────────────────────

export default function SMERiskDashboard() {
  const [activeTab, setActiveTab]     = useState("overview");
  const [audienceView, setAudienceView] = useState("all");
  const [activeSector, setActiveSector] = useState(null);
  const [activeElIdx, setActiveElIdx]   = useState(1);
  const [activeFiscalIdx, setActiveFiscalIdx] = useState(1);
  const [activeRec, setActiveRec]       = useState("ncgcl");

  const tabs = [
    { id: "overview",        label: "Overview",        icon: "◈" },
    { id: "portfolio",       label: "Portfolio",        icon: "◉" },
    { id: "sectors",         label: "Sectors",          icon: "◧" },
    { id: "institutions",    label: "Channels & Types", icon: "⊞" },
    { id: "systemic",        label: "System Risk",      icon: "⬡" },
    { id: "risk",            label: "Risk & EL",        icon: "◻" },
    { id: "fiscal",          label: "Fiscal Impact",    icon: "▣" },
    { id: "alerts",          label: "Early Warnings",   icon: "⚠" },
    { id: "annex",           label: "Annex A",          icon: "≡" },
    { id: "recommendations", label: "Actions",          icon: "▸" },
  ];

  const audiences = [
    { id: "all",       label: "All Audiences" },
    { id: "ceo",       label: "Banking CEO" },
    { id: "policy",    label: "Policy Maker" },
    { id: "regulator", label: "Regulator" },
  ];

  const renderContent = () => {
    const props = { activeSector, setActiveSector, activeElIdx, setActiveElIdx, activeFiscalIdx, setActiveFiscalIdx, activeRec, setActiveRec };
    switch (activeTab) {
      case "overview":        return <OverviewTab />;
      case "portfolio":       return <PortfolioTab />;
      case "sectors":         return <SectorsTab {...props} />;
      case "institutions":    return <InstitutionsTab />;
      case "systemic":        return <SystemicTab />;
      case "risk":            return <RiskTab {...props} />;
      case "fiscal":          return <FiscalTab {...props} />;
      case "alerts":          return <AlertsTab />;
      case "annex":           return <AnnexTab />;
      case "recommendations": return <RecommendationsTab {...props} />;
      default: return null;
    }
  };

  return (
    <div style={{ fontFamily: "'Lora','Georgia',serif", background: A.surfaceWarm, minHeight: "100vh", color: A.text }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Lora:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-thumb { background: ${A.stone}; border-radius: 2px; }
        ::-webkit-scrollbar-track { background: transparent; }
        @keyframes fadeUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
        .fi { animation: fadeUp 0.28s ease; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
        .pulse { animation: pulse 2.2s infinite; }
        table { border-collapse: collapse; width: 100%; }
        th { white-space: nowrap; }
        .tab-btn:hover { color: ${A.dark} !important; }
        .card-hover { transition: box-shadow 0.2s, transform 0.2s; }
        .card-hover:hover { box-shadow: 0 6px 24px rgba(20,20,19,0.09) !important; transform: translateY(-1px); }
        .rec-btn:hover { opacity: 0.88; }
        .audience-btn:hover { opacity: 0.85; }
        .sector-row:hover { background: ${A.cream} !important; }
      `}</style>

      {/* ── HEADER ── warm dark background = Anthropic #141413 wrapping NCGCL identity */}
      <div style={{ background: A.dark }}>
        <div style={{ maxWidth: 1280, margin: "0 auto", padding: "0 28px" }}>

          {/* Top strip */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "20px 0 16px", flexWrap: "wrap", gap: 10 }}>
            <NCGCLLogo size={40} />
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 9.5, color: "rgba(250,249,245,0.4)", fontFamily: "'Poppins',sans-serif", letterSpacing: "0.12em", textTransform: "uppercase" }}>Restricted — MoF &amp; Board Use Only</div>
              <div style={{ fontSize: 11, color: "rgba(250,249,245,0.55)", fontFamily: "'Lora','Georgia',serif", fontStyle: "italic", marginTop: 2 }}>SBP/SME-RAD/2026-04 · Updated February 20, 2026</div>
              <div style={{ fontSize: 10, color: "rgba(250,249,245,0.35)", fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>Data: ISIC Table 3.14 (Dec-25) · Table 1.9 (Dec-25) · SBP SME (Jan-26)</div>
            </div>
          </div>

          {/* Title block with Anthropic slash motif */}
          <div style={{ borderTop: `1px solid rgba(250,249,245,0.1)`, paddingTop: 18, paddingBottom: 0 }}>
            <div style={{ fontSize: 11, fontFamily: "'Poppins',sans-serif", fontWeight: 500, color: A.orange, letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: 8 }}>
              Risk Assessment
              <Slash color={A.orange} size={11} />
              Guarantee Scheme Liability Forecast
              <Slash color={A.orange} size={11} />
              v2.0
            </div>
            <div style={{ fontSize: 24, fontWeight: 600, color: A.light, fontFamily: "'Poppins',sans-serif", lineHeight: 1.25, letterSpacing: "-0.01em", maxWidth: 700 }}>
              SME Financing Portfolio<br />
              <span style={{ color: "rgba(250,249,245,0.6)", fontWeight: 400, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic", fontSize: 18 }}>Comprehensive update with SBP Dec-25 data integration</span>
            </div>

            {/* Audience selector — clean pill buttons */}
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 18, paddingBottom: 0 }}>
              {audiences.map(a => (
                <button key={a.id} className="audience-btn" onClick={() => setAudienceView(a.id)} style={{
                  background: audienceView === a.id ? A.cerulean : "rgba(250,249,245,0.07)",
                  color: audienceView === a.id ? A.light : "rgba(250,249,245,0.5)",
                  border: `1px solid ${audienceView === a.id ? A.cerulean : "rgba(250,249,245,0.15)"}`,
                  borderRadius: "20px 20px 0 0",
                  padding: "7px 18px",
                  cursor: "pointer", fontSize: 11, fontWeight: 500, fontFamily: "'Poppins',sans-serif",
                  letterSpacing: "0.04em", transition: "all 0.18s"
                }}>{a.label}</button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── AUDIENCE STRIP ── warm cerulean tint */}
      {audienceView !== "all" && audienceMessages[audienceView] && (
        <div style={{ background: A.cerulean, padding: "14px 28px", borderBottom: `1px solid ${A.cyan}` }}>
          <div style={{ maxWidth: 1280, margin: "0 auto" }}>
            <div style={{ color: A.light, fontWeight: 600, fontSize: 13.5, fontFamily: "'Poppins',sans-serif", marginBottom: 9 }}>
              {audienceMessages[audienceView].headline}
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
              {audienceMessages[audienceView].bullets.map((b, i) => (
                <div key={i} style={{
                  background: "rgba(250,249,245,0.1)", borderRadius: 6,
                  padding: "5px 12px", fontSize: 11.5,
                  color: "rgba(250,249,245,0.88)", fontFamily: "'Lora','Georgia',serif",
                  maxWidth: 380, border: "1px solid rgba(250,249,245,0.12)"
                }}>
                  <span style={{ color: A.green, fontWeight: 700, marginRight: 6 }}>›</span>{b}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── NAV TABS ── white bar with Anthropic warm border */}
      <div style={{ background: A.surface, borderBottom: `1.5px solid ${A.border}`, position: "sticky", top: 0, zIndex: 100, boxShadow: "0 1px 4px rgba(20,20,19,0.06)" }}>
        <div style={{ maxWidth: 1280, margin: "0 auto", padding: "0 28px", display: "flex", overflowX: "auto", gap: 0 }}>
          {tabs.map(t => (
            <button key={t.id} className="tab-btn" onClick={() => setActiveTab(t.id)} style={{
              background: "transparent", border: "none",
              padding: "11px 15px",
              cursor: "pointer",
              fontSize: 11.5,
              fontWeight: activeTab === t.id ? 600 : 400,
              color: activeTab === t.id ? A.cerulean : A.textMid,
              fontFamily: "'Poppins',sans-serif",
              letterSpacing: "0.02em",
              borderBottom: activeTab === t.id ? `2.5px solid ${A.cerulean}` : "2.5px solid transparent",
              whiteSpace: "nowrap", transition: "color 0.15s",
            }}>
              <span style={{ marginRight: 5, fontSize: 10, opacity: 0.5 }}>{t.icon}</span>{t.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── CONTENT ── */}
      <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 28px 80px" }}>
        <div className="fi" key={activeTab}>{renderContent()}</div>
      </div>

      {/* ── FOOTER ── Anthropic dark again, NCGCL credits */}
      <div style={{ background: A.dark, padding: "20px 28px", borderTop: `1px solid rgba(250,249,245,0.07)` }}>
        <div style={{ maxWidth: 1280, margin: "0 auto", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <NCGCLLogo height={22} />
            <span style={{ color: "rgba(250,249,245,0.4)", fontSize: 10.5, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>
              © 2026 Risk &amp; Analytics Division
            </span>
          </div>
          <div style={{ color: "rgba(250,249,245,0.3)", fontSize: 10, fontFamily: "'Poppins',sans-serif", letterSpacing: "0.04em" }}>
            Sources: SBP Table 3.14 (Dec-25) · Table 1.9 (Dec-25) · SME Snapshot (Sep-25) · Monetary Statistics (Jan-26)
            <Slash color="rgba(250,249,245,0.2)" size={9} />
            Classified Restricted · v2.0 · Feb 20, 2026
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── SHARED CARD WRAPPER ─────────────────────────────────────────────────────
const Card = ({ children, style = {}, className = "" }) => (
  <div className={`card-hover ${className}`} style={{
    background: A.surface,
    borderRadius: 10,
    border: `1px solid ${A.border}`,
    boxShadow: "0 1px 5px rgba(20,20,19,0.05)",
    padding: 22,
    ...style
  }}>{children}</div>
);

const CardTitle = ({ children, badge }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
    <div style={{ fontSize: 13.5, fontWeight: 600, color: A.dark, fontFamily: "'Poppins',sans-serif", lineHeight: 1.3 }}>{children}</div>
    {badge && <span style={{ background: "#f0f8f2", color: A.green, padding: "2px 8px", borderRadius: 20, fontSize: 9.5, fontWeight: 600, fontFamily: "'Poppins',sans-serif", border: `1px solid ${A.green}30` }}>{badge}</span>}
  </div>
);

const CardSub = ({ children }) => (
  <div style={{ fontSize: 11, color: A.textLight, marginBottom: 14, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>{children}</div>
);

const TH = ({ children, left = false }) => (
  <th style={{ background: A.cerulean, color: A.light, padding: "9px 11px", textAlign: left ? "left" : "center", fontFamily: "'Poppins',sans-serif", fontSize: 10, fontWeight: 600, letterSpacing: "0.04em" }}>{children}</th>
);

// ─── OVERVIEW TAB ─────────────────────────────────────────────────────────────
function OverviewTab() {
  return (
    <div>
      {/* KPI Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(185px,1fr))", gap: 14, marginBottom: 28 }}>
        <StatCard label="Dec-25 Outstanding" value="PKR 947.9 Bn" sub="ISIC Table 3.14 · +66.3% since Jun-24" color={A.cerulean} size="lg" />
        <StatCard label="Jan-26 Outstanding" value="PKR 937.0 Bn" sub="−1.1% MoM post year-end surge" color={A.cerulean} size="lg" />
        <StatCard label="YoY Growth (Dec-25)" value="+47.5%" sub="vs total private sector +4.9%" color={A.green} size="lg" />
        <StatCard label="H2-2025 Addition" value="PKR 191.6 Bn" sub="+25.3% in just 6 months" color={A.blue} size="lg" />
        <StatCard label="Active Borrowers" value="295,291" sub="Sep-25 · +66.2% YoY · avg PKR 2.32M" color={A.cerulean} size="lg" />
        <StatCard label="Agriculture NPL Ratio" value="15.3%" sub="↑ from 9.8% Dec-24 · NPLs +80.8% abs." color={A.red} warning size="lg" />
      </div>

      {/* Main chart row */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 20, marginBottom: 22 }}>
        <Card>
          <CardTitle>Portfolio Trajectory — Jun-24 to Jan-26</CardTitle>
          <CardSub>PKR Billions · ISIC Table 3.14 (Dec-25 cut-off) + SBP SME Jan-26</CardSub>
          <SparkLine data={timeSeriesData} color={A.cerulean} height={82} />
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, marginBottom: 16 }}>
            <span style={{ fontSize: 10, color: A.textLight, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>Jun-24: 570.1 Bn</span>
            <span style={{ fontSize: 10, color: A.orange, fontWeight: 600, fontFamily: "'Poppins',sans-serif" }}>Dec-25 surge: +PKR 90.4 Bn ⚠</span>
            <span style={{ fontSize: 10, color: A.cerulean, fontWeight: 600, fontFamily: "'Poppins',sans-serif" }}>Jan-26: 937.0 Bn</span>
          </div>
          <div style={{ display: "flex", gap: 14, paddingTop: 14, borderTop: `1px solid ${A.border}` }}>
            {[["Phase 1","Jun-24 → Jun-25","FY25: +32.7%",A.green],["Phase 2","Jul-25 → Nov-25","+5–6% MoM",A.blue],["Phase 3","Dec-25","Year-End: +10.5% ⚠",A.orange]].map(([p,d,n,c])=>(
              <div key={p} style={{ flex: 1 }}>
                <div style={{ width: 20, height: 3, background: c, borderRadius: 2, marginBottom: 5 }} />
                <div style={{ fontSize: 11, fontWeight: 600, color: c, fontFamily: "'Poppins',sans-serif" }}>{p}</div>
                <div style={{ fontSize: 10, color: A.textLight, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>{d}</div>
                <div style={{ fontSize: 10.5, color: A.text, fontFamily: "'Lora','Georgia',serif" }}>{n}</div>
              </div>
            ))}
          </div>
        </Card>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Card>
            <CardTitle>SME vs Agriculture NPL Ratio</CardTitle>
            <div style={{ display: "flex", justifyContent: "space-around", marginTop: 8 }}>
              <Gauge value={12.5} max={25} label="SME (Dec-25)" color={A.orange} size={82} />
              <Gauge value={15.3} max={25} label="Agri (Dec-25)" color={A.red} size={82} />
            </div>
            <AlertBox type="critical">Agriculture infection now <strong>exceeds SME</strong>. NPLs +80.8% absolute (Dec-24 → Dec-25).</AlertBox>
          </Card>
          <Card>
            <CardTitle>Credit Reallocation (Dec-24→Dec-25)</CardTitle>
            <div style={{ marginTop: 10 }}>
              <HBar label="SME Advances" value="+33.2%" max={40} color={A.green} sub="PKR 678 → 903 Bn" />
              <HBar label="Agriculture" value="+16.4%" max={40} color={A.orange} sub="PKR 578 → 674 Bn" />
              <HBar label="Consumer" value="+15.8%" max={40} color={A.blue} sub="PKR 891 → 1,032 Bn" />
              <HBar label="Corporate" value="-11.4%" max={40} color={A.red} negative sub="PKR 12,305 → 10,897 Bn — contracting" />
            </div>
            <div style={{ fontSize: 10.5, color: A.orange, fontWeight: 600, fontFamily: "'Poppins',sans-serif", marginTop: 4 }}>PKR 1.4 Trillion exiting corporate; flowing to SME</div>
          </Card>
        </div>
      </div>

      {/* Risk signals grid */}
      <Card style={{ marginBottom: 22 }}>
        <CardTitle>Critical Risk Signals — Updated Dec-25 Data</CardTitle>
        <div style={{ marginBottom: 14 }} />
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(268px,1fr))", gap: 10 }}>
          <AlertBox type="critical" title="Dec-25 Year-End Spike — Verify">+PKR 90.4 Bn (+10.5%) in a single month. 85% from 3 sectors; 63% from working capital. Likely seasonal drawdown — validate before using Dec-25 as EAD baseline.</AlertBox>
          <AlertBox type="critical" title="Agriculture Alarm — 15.3% NPL">Agriculture infection ratio now HIGHEST among major segments. NPLs rose +80.8% absolute (Dec-24→Dec-25). Systemic correlation with Indus Basin cycle.</AlertBox>
          <AlertBox type="warning" title="Data Reconciliation Required">Three SBP tables give different SME exposure figures. ISIC 3.14 ≠ Table 1.9 ≠ Sep SME Snapshot. No policy figure should be final without a reconciliation map.</AlertBox>
          <AlertBox type="warning" title="Ticket Compression: −13.6%">Avg outstanding per borrower: PKR 2.69M → 2.32M (Sep-24→Sep-25). More borrowers, smaller loans = higher admin cost and default frequency per PKR.</AlertBox>
          <AlertBox type="warning" title="Fixed Investment Share: 49.6%">Fixed Investment now nearly half the SME book. Loans from 2022-23 reaching 3-year refinancing at sustained high rates — hidden NPL risk.</AlertBox>
          <AlertBox type="info" title="Corporate Credit Contracting">Corporate advances fell PKR 1.4 Trillion (−11.4%). SME growing +33.2%. Structural reallocation — policy intent vs. credit discipline tension.</AlertBox>
        </div>
      </Card>

      {/* Concentration bars */}
      <Card>
        <CardTitle>Portfolio Concentration — Top-5 Sectors = 90.1% of Total (Dec-25)</CardTitle>
        <CardSub>Wholesale/Retail + Manufacturing = 67.8% · Extreme two-sector concentration</CardSub>
        <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
          {sectorData.slice(0,5).map(s => (
            <div key={s.sector} style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ width: 8, height: 8, background: s.color, borderRadius: 2, flexShrink: 0 }} />
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                  <span style={{ fontSize: 12, fontFamily: "'Lora','Georgia',serif", color: A.text }}>{s.sector}</span>
                  <div style={{ display: "flex", gap: 14 }}>
                    <span style={{ fontSize: 11, fontWeight: 600, color: s.color, fontFamily: "'Poppins',sans-serif" }}>{s.share}%</span>
                    <span style={{ fontSize: 11, color: A.green, fontFamily: "'Poppins',sans-serif" }}>+{s.growth}%</span>
                    <span style={{ fontSize: 11, color: A.textLight, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>{s.contrib > 0 ? "+" : ""}{s.contrib}% of growth</span>
                  </div>
                </div>
                <div style={{ height: 5, background: A.lightGray, borderRadius: 3 }}>
                  <div style={{ height: "100%", width: `${s.share * 2.6}%`, background: s.color, borderRadius: 3 }} />
                </div>
              </div>
              <span style={{ fontSize: 11.5, fontWeight: 600, color: A.cerulean, minWidth: 72, textAlign: "right", fontFamily: "'Poppins',sans-serif" }}>PKR {s.dec25} Bn</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ─── PORTFOLIO TAB ────────────────────────────────────────────────────────────
function PortfolioTab() {
  return (
    <div>
      <SectionHeader number="1" title="Portfolio Dynamics & Borrower Analysis" subtitle="ISIC Table 3.14 · Jun-24 to Jan-26 · Q4-2025 MoM breakdown · Borrower & ticket data" />

      <Card style={{ marginBottom: 22, overflowX: "auto" }}>
        <CardTitle>Table 1 — SME Financing Outstanding (ISIC Table 3.14)</CardTitle>
        <div style={{ height: 12 }} />
        <table>
          <thead>
            <tr>
              {["Period","Outstanding (PKR Bn)","Period Growth","MoM Growth","Phase","Note"].map((h,i) => <TH key={h} left={i===0||i===5}>{h}</TH>)}
            </tr>
          </thead>
          <tbody>
            {timeSeriesData.map((r, i) => {
              const pc = [A.sage, A.blue, A.orange][r.phase - 1];
              return (
                <tr key={i} style={{ background: i % 2 === 0 ? A.surfaceWarm : A.surface, borderBottom: `1px solid ${A.border}` }}>
                  <td style={{ padding: "8px 11px", fontWeight: 600, color: A.cerulean, fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{r.period}</td>
                  <td style={{ padding: "8px 11px", textAlign: "center", fontWeight: 700, color: A.dark, fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{r.outstanding}</td>
                  <td style={{ padding: "8px 11px", textAlign: "center", color: r.growth > 0 ? A.green : A.red, fontWeight: 600, fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{r.growth !== null ? (r.growth > 0 ? "+" : "") + r.growth + "%" : "—"}</td>
                  <td style={{ padding: "8px 11px", textAlign: "center", color: r.mom > 8 ? A.orange : A.textMid, fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{r.mom !== null ? (r.mom > 0 ? "+" : "") + r.mom + "%" : "—"}</td>
                  <td style={{ padding: "8px 11px", textAlign: "center" }}><span style={{ background: `${pc}18`, color: pc, padding: "2px 9px", borderRadius: 12, fontSize: 10, fontWeight: 600, fontFamily: "'Poppins',sans-serif" }}>Phase {r.phase}</span></td>
                  <td style={{ padding: "8px 11px", fontSize: 11, color: r.note.includes("⚠") ? A.orange : A.textLight, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>{r.note}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>

      {/* Q4 MoM */}
      <Card style={{ marginBottom: 22 }}>
        <CardTitle badge="★ New">Table 6 — Q4-2025 Month-on-Month Growth Drivers</CardTitle>
        <CardSub>Sector contribution to each month's MoM change · ISIC Table 3.14</CardSub>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14 }}>
          {q4momData.map((m, i) => (
            <div key={i} style={{
              background: m.alert ? "#fef8f4" : A.surfaceWarm,
              borderRadius: 9, padding: 18,
              border: `1.5px solid ${m.alert ? A.orange : A.border}`,
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: m.alert ? A.orange : A.dark, fontFamily: "'Poppins',sans-serif", marginBottom: 4 }}>{m.period}</div>
              <div style={{ fontSize: 28, fontWeight: 700, color: m.alert ? A.orange : A.cerulean, fontFamily: "'Poppins',sans-serif", lineHeight: 1, letterSpacing: "-0.02em" }}>+{m.mom}%</div>
              <div style={{ fontSize: 11.5, color: A.textMid, marginBottom: 13, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>+PKR {m.delta} Bn MoM</div>
              {m.top.map(([sect, pct]) => (
                <div key={sect} style={{ marginBottom: 8 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                    <span style={{ fontSize: 10.5, color: A.text, fontFamily: "'Lora','Georgia',serif" }}>{sect}</span>
                    <span style={{ fontSize: 10.5, fontWeight: 600, color: m.alert ? A.orange : A.cerulean, fontFamily: "'Poppins',sans-serif" }}>{pct}%</span>
                  </div>
                  <div style={{ height: 4, background: A.lightGray, borderRadius: 2 }}>
                    <div style={{ height: "100%", width: `${pct}%`, background: m.alert ? A.orange : A.cerulean, borderRadius: 2 }} />
                  </div>
                </div>
              ))}
              {m.alert && <AlertBox type="warning">Dec-25 surge: 63% WC-driven. Warrants data reconciliation — likely seasonal, not durable acceleration.</AlertBox>}
            </div>
          ))}
        </div>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        <Card>
          <CardTitle badge="★ Updated">Table 3 — Borrower Count &amp; Ticket Size</CardTitle>
          <CardSub>Sep SBP SME dataset · Banks &amp; DFIs</CardSub>
          <table>
            <thead>
              <tr>{["Date","Outstanding (Bn)","Borrowers","Avg Ticket (Mn)"].map((h,i) => <TH key={h} left={i===0}>{h}</TH>)}</tr>
            </thead>
            <tbody>
              {borrowerData.map((r, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? A.surfaceWarm : A.surface, borderBottom: `1px solid ${A.border}` }}>
                  <td style={{ padding: "8px 11px", fontWeight: 600, color: A.cerulean, fontFamily: "'Poppins',sans-serif" }}>{r.period}</td>
                  <td style={{ padding: "8px 11px", textAlign: "center", fontFamily: "'Poppins',sans-serif" }}>{r.outstanding}</td>
                  <td style={{ padding: "8px 11px", textAlign: "center", fontWeight: 600, color: A.green, fontFamily: "'Poppins',sans-serif" }}>{r.borrowers.toLocaleString()}</td>
                  <td style={{ padding: "8px 11px", textAlign: "center", fontWeight: 700, color: r.avgTicket < 2.5 ? A.orange : A.cerulean, fontFamily: "'Poppins',sans-serif" }}>PKR {r.avgTicket}M</td>
                </tr>
              ))}
            </tbody>
          </table>
          <AlertBox type="warning">Avg ticket compressed −13.6% (Sep-24→Sep-25). Borrower growth (+66.2%) outpaced outstanding growth (+43.4%) — structural downtiering to smaller, riskier borrowers.</AlertBox>
        </Card>

        <Card>
          <CardTitle badge="★ New">Table 5 — Facility Mix Shift (Sep-24 vs Sep-25)</CardTitle>
          <CardSub>Shift toward investment-led lending · fixed investment now dominant</CardSub>
          {facilityData.map(f => (
            <div key={f.type} style={{ marginBottom: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontSize: 12.5, fontFamily: "'Lora','Georgia',serif", color: A.text }}>{f.type}</span>
                <span style={{ fontSize: 11, fontWeight: 600, color: f.change > 0 ? A.cerulean : A.midGray, fontFamily: "'Poppins',sans-serif" }}>{f.change > 0 ? "+" : ""}{f.change}pp</span>
              </div>
              <div style={{ height: 9, background: A.lightGray, borderRadius: 5, overflow: "hidden", position: "relative" }}>
                <div style={{ position: "absolute", top: 0, left: 0, height: "100%", width: `${f.shareSep24}%`, background: A.stone, borderRadius: 5, opacity: 0.5 }} />
                <div style={{ position: "absolute", top: 0, left: 0, height: "100%", width: `${f.shareSep25}%`, background: f.change > 0 ? A.cerulean : A.midGray, borderRadius: 5, opacity: 0.75 }} />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10.5, color: A.textLight, marginTop: 3, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>
                <span>Sep-24: {f.shareSep24}% (PKR {f.sep24} Bn)</span>
                <span style={{ fontWeight: 500, color: f.change > 0 ? A.cerulean : A.midGray }}>Sep-25: {f.shareSep25}% (PKR {f.sep25} Bn)</span>
              </div>
            </div>
          ))}
          <AlertBox type="warning">Fixed Investment at 49.6% creates refinancing cliff risk. FI loans from 2022-23 now at 3-year maturity at sustained high KIBOR — potential hidden NPL formation.</AlertBox>
        </Card>
      </div>
    </div>
  );
}

// ─── SECTORS TAB ─────────────────────────────────────────────────────────────
function SectorsTab({ activeSector, setActiveSector }) {
  const sectorNotes = {
    "Wholesale & Retail Trade": { desc: "Largest sector, but dominated by Working Capital (PKR 178.0 Bn = 54.6%) — high rollover risk. At KIBOR 10.5%, interest burden on low-margin retailers is at absorptive limit.", key: "Single largest NPL exposure source. WC rollover denial = liquidity crisis." },
    "Manufacturing": { desc: "Stable anchor with tangible collateral. Food Products and Rice Processing dominate. WC dominant (PKR 215.1 Bn). Textile sub-sector facing Utility Tariff Cliff.", key: "30% energy tariff hike impairs PKR 85 Bn of manufacturing SME cash flows." },
    "Agriculture & Fishing": { desc: "+91.7% growth in 18 months. System-wide agriculture NPL ratio jumped to 15.3% (from 9.8%) — now highest among all banking segments. NPLs rose +80.8%.", key: "Agriculture infection ratio now exceeds SME. One bad season = PKR 95 Bn simultaneous impairment." },
    "Other Service Activities": { desc: "153.2% growth — fastest growing major sector. ISIC 95 (beauty parlors, tailoring, repair). Near-zero collateral, cash-based, high failure rates.", key: "Highest risk per PKR of outstanding. EL disproportionate to book size." },
    "Transportation & Storage": { desc: "Asset-backed (vehicles, fleet) — better LGD profile. FI-dominant (PKR 32.3 Bn of PKR 56.2 Bn). Counter-trend growth signal in late 2025.", key: "Physical assets recoverable. Moderate risk relative to size." },
    "Construction": { desc: "Moderate growth (+41.3%). Balanced WC/FI split. Real estate cycle exposure and completion risk.", key: "Real estate cycle downturn could elevate NPLs rapidly." },
    "Professional & Scientific": { desc: "UNIQUE: Only sector that DECLINED (−13.9%). Could signal de-risking, definitional reclassification, or genuine demand contraction.", key: "Anomalous contraction — regulatory data quality signal." },
    "Admin & Support Services": { desc: "+55.6% growth. PKR 12.7 Bn. Includes staffing, security, cleaning. Generally collateral-light.", key: "Growing but manageable. Monitor FI portion for refinancing risk." },
    "ICT": { desc: "+128.9% growth. FI-heavy (PKR 4.8 Bn). Technology assets have rapid obsolescence — LGD likely higher than assumed.", key: "Infrastructure capex in illiquid assets. Real LGD may be 80–90% on FI portion." },
    "Accommodation & Food": { desc: "+106.3% growth. High sector failure rate. Location-dependent, informal-sector heavy. FI-dominant (PKR 5.0 Bn).", key: "Hospitality defaults spike sharply in demand downturns." },
  };

  return (
    <div>
      <SectionHeader number="2" title="Sector Deep Dive — Dec-25 Precise Figures" subtitle="ISIC Table 3.14 · Dec-25 outstanding · Jun-24→Dec-25 growth · Contribution to growth" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 22 }}>
        <Card>
          <CardTitle>Table 4 — Sector Composition &amp; Growth</CardTitle>
          <CardSub>Click a sector to explore risk profile and financing mix</CardSub>
          {sectorData.map(s => (
            <div key={s.sector} className="sector-row" onClick={() => setActiveSector(activeSector === s.sector ? null : s.sector)}
              style={{
                display: "flex", alignItems: "center", gap: 10, marginBottom: 6,
                cursor: "pointer", padding: "8px 10px", borderRadius: 7,
                background: activeSector === s.sector ? `${s.color}12` : "transparent",
                border: `1px solid ${activeSector === s.sector ? s.color + "50" : "transparent"}`,
                transition: "background 0.15s, border 0.15s",
              }}>
              <div style={{ width: 8, height: 8, background: s.color, borderRadius: 2, flexShrink: 0 }} />
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                  <span style={{ fontSize: 12, fontFamily: "'Lora','Georgia',serif", color: A.text }}>{s.sector}</span>
                  <span style={{ fontSize: 11, fontWeight: 600, color: s.color, fontFamily: "'Poppins',sans-serif" }}>{s.share}%</span>
                </div>
                <div style={{ height: 4, background: A.lightGray, borderRadius: 2 }}>
                  <div style={{ height: "100%", width: `${s.share * 2.6}%`, background: s.color, borderRadius: 2 }} />
                </div>
              </div>
              <div style={{ textAlign: "right", minWidth: 96 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: A.dark, fontFamily: "'Poppins',sans-serif" }}>PKR {s.dec25} Bn</div>
                <div style={{ fontSize: 10, color: s.growth < 0 ? A.red : A.green, fontFamily: "'Poppins',sans-serif" }}>{s.growth > 0 ? "+" : ""}{s.growth}%</div>
              </div>
              <RiskBadge level={s.risk} />
            </div>
          ))}
        </Card>

        {activeSector ? (() => {
          const s = sectorData.find(x => x.sector === activeSector);
          const nd = sectorNotes[s.sector] || { desc: "Emerging sector.", key: "Monitor for concentration." };
          const wc = (s.wc / 1000).toFixed(1), fi = (s.fi / 1000).toFixed(1), tf = (s.tfin / 1000).toFixed(1);
          return (
            <Card className="fi" style={{ border: `1.5px solid ${s.color}40` }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
                <div style={{ fontSize: 15, fontWeight: 600, color: s.color, fontFamily: "'Poppins',sans-serif" }}>{s.sector}</div>
                <RiskBadge level={s.risk} />
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10, marginBottom: 16 }}>
                {[["Dec-25", `PKR ${s.dec25} Bn`, s.color], [`${s.growth < 0 ? "" : "+"}${s.growth}%`, "Growth", s.growth < 0 ? A.red : A.green], [`${s.contrib > 0 ? "+" : ""}${s.contrib}%`, "Contrib.", s.contrib < 0 ? A.red : A.cerulean]].map(([val, lab, col]) => (
                  <div key={lab} style={{ textAlign: "center", padding: 12, background: `${col}0e`, borderRadius: 8, border: `1px solid ${col}25` }}>
                    <div style={{ fontSize: 17, fontWeight: 700, color: col, fontFamily: "'Poppins',sans-serif", lineHeight: 1 }}>{val}</div>
                    <div style={{ fontSize: 10, color: A.textLight, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic", marginTop: 3 }}>{lab}</div>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 12, color: A.text, lineHeight: 1.7, marginBottom: 14, fontFamily: "'Lora','Georgia',serif" }}>{nd.desc}</div>
              <div style={{ fontSize: 12, fontWeight: 600, color: A.dark, fontFamily: "'Poppins',sans-serif", marginBottom: 8 }}>Financing Mix (Dec-25, PKR Bn)</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 8, marginBottom: 14 }}>
                {[["Trade Finance", tf, A.green], ["Working Capital", wc, A.cerulean], ["Fixed Investment", fi, A.orange]].map(([t, v, c]) => (
                  <div key={t} style={{ textAlign: "center", background: `${c}0e`, borderRadius: 6, padding: 8, border: `1px solid ${c}20` }}>
                    <div style={{ fontSize: 14, fontWeight: 700, color: c, fontFamily: "'Poppins',sans-serif" }}>PKR {v} Bn</div>
                    <div style={{ fontSize: 9.5, color: A.textLight, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>{t}</div>
                  </div>
                ))}
              </div>
              <div style={{ padding: "9px 13px", background: `${s.color}0d`, borderLeft: `3px solid ${s.color}`, borderRadius: "0 6px 6px 0" }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: s.color, fontFamily: "'Poppins',sans-serif" }}>Key Risk: </span>
                <span style={{ fontSize: 11.5, color: A.text, fontFamily: "'Lora','Georgia',serif" }}>{nd.key}</span>
              </div>
            </Card>
          );
        })() : (
          <div style={{ background: A.surfaceWarm, borderRadius: 10, border: `1px solid ${A.border}`, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 10 }}>
            <div style={{ fontSize: 28, opacity: 0.15 }}>◧</div>
            <div style={{ fontSize: 12, color: A.textLight, textAlign: "center", fontFamily: "'Lora','Georgia',serif", fontStyle: "italic", padding: 24 }}>Click a sector to explore its risk profile, financing mix, and key vulnerabilities</div>
          </div>
        )}
      </div>

      <Card style={{ overflowX: "auto" }}>
        <CardTitle badge="★ New">Sector Contribution to SME Growth (Jun-24→Dec-25)</CardTitle>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 10, marginTop: 14 }}>
          {sectorData.map(s => (
            <div key={s.code} style={{
              background: s.contrib < 0 ? "#fdf0f0" : A.surfaceWarm,
              borderRadius: 9, padding: "12px 13px",
              border: `1px solid ${s.contrib < 0 ? A.red + "40" : s.color + "30"}`,
              borderLeft: `3px solid ${s.contrib < 0 ? A.red : s.color}`,
              display: "flex", flexDirection: "column", gap: 4,
            }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: s.color, fontFamily: "'Poppins',sans-serif", letterSpacing: "0.06em", textTransform: "uppercase" }}>ISIC {s.code}</div>
              <div style={{ fontSize: 11.5, fontFamily: "'Lora','Georgia',serif", color: A.text, lineHeight: 1.35 }}>{s.sector}</div>
              <div style={{ display: "flex", alignItems: "baseline", gap: 6, marginTop: 2 }}>
                <span style={{ fontSize: 22, fontWeight: 700, color: s.contrib < 0 ? A.red : s.color, fontFamily: "'Poppins',sans-serif", lineHeight: 1, letterSpacing: "-0.02em" }}>{s.contrib > 0 ? "+" : ""}{s.contrib}%</span>
                <span style={{ fontSize: 10, color: A.textLight, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>of growth</span>
              </div>
              <div style={{ fontSize: 10.5, color: s.growth < 0 ? A.red : A.textMid, fontFamily: "'Poppins',sans-serif" }}>{s.growth > 0 ? "+" : ""}{s.growth}% sector growth</div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 14 }}>
          <AlertBox type="warning" title="Professional & Scientific (L): Only negative contributor">Sector L declined −13.9% since Jun-24, contributing −0.7% to total SME growth. Unique anomaly — investigate: policy-driven de-risking, definitional change, or genuine credit contraction.</AlertBox>
        </div>
      </Card>
    </div>
  );
}

// ─── INSTITUTIONS & TYPES TAB ─────────────────────────────────────────────────
function InstitutionsTab() {
  return (
    <div>
      <SectionHeader number="3" title="Banking Channels, SME Types & Facility Mix" subtitle="Table 7 · Sep-24 vs Sep-25 · New dataset revealing structural shifts in who lends and to whom" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 22 }}>
        <Card>
          <CardTitle badge="★ New">Table 7a — SME Outstanding by Banking Channel</CardTitle>
          <CardSub>Sep-24 → Sep-25 · Share and growth dynamics reveal program-driven distortions</CardSub>
          {bankingData.map(b => (
            <div key={b.channel} style={{
              marginBottom: 12, padding: "10px 13px",
              background: b.trend === "rising" ? "#fef8f4" : b.trend === "declining" ? "#fdf0f0" : A.surfaceWarm,
              borderRadius: 8, border: `1px solid ${b.color}25`,
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 5 }}>
                <div style={{ fontSize: 12.5, fontFamily: "'Lora','Georgia',serif", color: A.dark }}>{b.channel}</div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 15, fontWeight: 700, color: b.growth < 0 ? A.red : b.growth > 60 ? A.orange : A.green, fontFamily: "'Poppins',sans-serif" }}>{b.growth > 0 ? "+" : ""}{b.growth}%</div>
                  <div style={{ fontSize: 10, color: A.textLight, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>Sep-25 share: {b.shareSep25}%</div>
                </div>
              </div>
              <div style={{ display: "flex", gap: 14, fontSize: 11, color: A.textMid, fontFamily: "'Lora','Georgia',serif" }}>
                <span>Sep-24: PKR {b.sep24} Bn</span>
                <span style={{ color: b.growth > 0 ? A.cerulean : A.red, fontWeight: 500 }}>Sep-25: PKR {b.sep25} Bn</span>
              </div>
            </div>
          ))}
          <AlertBox type="critical" title="Public Sector Banks: +71.3% — Underwriting Risk">Public sector banks growing fastest (+71.3%) with weakest historical underwriting. Share rising 26.9% → 32.1%. Program-driven targets likely overriding credit discipline.</AlertBox>
        </Card>

        <Card>
          <CardTitle badge="★ New">Table 7b — SME Outstanding by SME Type</CardTitle>
          <CardSub>Services SMEs fastest growing — but highest risk profile per unit of credit</CardSub>
          {smeTypeData.map(s => (
            <div key={s.type} style={{ marginBottom: 18 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontSize: 13, fontFamily: "'Poppins',sans-serif", fontWeight: 500, color: A.dark }}>{s.type}</span>
                <div>
                  <span style={{ fontSize: 14, fontWeight: 700, color: s.color, fontFamily: "'Poppins',sans-serif" }}>+{s.growth}%</span>
                  <span style={{ fontSize: 10, color: A.textLight, marginLeft: 8, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>{s.shareSep25}% share</span>
                </div>
              </div>
              <div style={{ height: 8, background: A.lightGray, borderRadius: 5 }}>
                <div style={{ height: "100%", width: `${s.shareSep25 * 2.5}%`, background: s.color, borderRadius: 5 }} />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10.5, color: A.textLight, marginTop: 3, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>
                <span>Sep-24: PKR {s.sep24} Bn</span>
                <span style={{ color: s.color, fontStyle: "normal", fontWeight: 500 }}>Sep-25: PKR {s.sep25} Bn</span>
              </div>
            </div>
          ))}
          <AlertBox type="warning" title="Services SMEs: +71.4% — Fastest Growing, Highest Risk">Services SMEs now 32.4% of borrower-based outstanding. Typically asset-light, cash-flow volatile, informal. Manufacturing SMEs growing slowest (+27.8%) — safest cohort losing relative share.</AlertBox>
          <div style={{ marginTop: 14, padding: 13, background: "#eef3f7", borderRadius: 8, border: `1px solid ${A.blue}30` }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: A.blue, fontFamily: "'Poppins',sans-serif", marginBottom: 5 }}>Islamic Banks: PKR 48.9 → 83.0 Bn (+69.7%)</div>
            <div style={{ fontSize: 11.5, color: A.text, fontFamily: "'Lora','Georgia',serif", lineHeight: 1.6 }}>Murabaha-based SME lending has different collateral profiles — separate NPL tracking protocol needed.</div>
          </div>
        </Card>
      </div>

      <Card style={{ overflowX: "auto" }}>
        <CardTitle>Full Table 7 — Comparative Summary (Sep-24 vs Sep-25)</CardTitle>
        <div style={{ height: 12 }} />
        <table>
          <thead>
            <tr>{["Category","Sep-24 (PKR Bn)","Sep-25 (PKR Bn)","Growth","Sep-25 Share","Signal"].map((h,i) => <TH key={h} left={i===0||i===5}>{h}</TH>)}</tr>
          </thead>
          <tbody>
            {[...smeTypeData.map(s => ({ cat: s.type, s24: s.sep24, s25: s.sep25, g: s.growth, sh: s.shareSep25, sig: s.growth > 60 ? "⚠ Fastest — monitor" : "Stable" })),
              ...bankingData.map(b => ({ cat: b.channel, s24: b.sep24, s25: b.sep25, g: b.growth, sh: b.shareSep25, sig: b.growth > 60 ? "⚠ Underwriting risk" : b.growth < 0 ? "↓ Contracting" : "Stable" }))
            ].map((r, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? A.surfaceWarm : A.surface, borderBottom: `1px solid ${A.border}` }}>
                <td style={{ padding: "8px 11px", fontWeight: 500, fontFamily: "'Lora','Georgia',serif", fontSize: 12 }}>{r.cat}</td>
                <td style={{ padding: "8px 11px", textAlign: "center", fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{r.s24.toFixed(2)}</td>
                <td style={{ padding: "8px 11px", textAlign: "center", fontWeight: 600, color: A.dark, fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{r.s25.toFixed(2)}</td>
                <td style={{ padding: "8px 11px", textAlign: "center", color: r.g < 0 ? A.red : r.g > 60 ? A.orange : A.green, fontWeight: 600, fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{r.g > 0 ? "+" : ""}{r.g}%</td>
                <td style={{ padding: "8px 11px", textAlign: "center", fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{r.sh}%</td>
                <td style={{ padding: "8px 11px", fontSize: 11.5, color: r.sig.includes("⚠") ? A.orange : A.textMid, fontFamily: "'Lora','Georgia',serif", fontStyle: r.sig.includes("⚠") ? "normal" : "italic" }}>{r.sig}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}

// ─── SYSTEMIC TAB ─────────────────────────────────────────────────────────────
function SystemicTab() {
  return (
    <div>
      <SectionHeader number="4" title="System-wide Segment Risk — Table 1.9 (Dec-25)" subtitle="Corporate · SME · Agriculture · Consumer · Advances and NPL infection ratios · Dec-24 vs Dec-25" />
      <AlertBox type="new" title="Entirely New Dataset — SBP Table 1.9 (Dec-25)">This section integrates SBP Table 1.9 Segment-wise Advances and NPLs through December 2025. It reveals a critical structural shift: corporate credit contracting while SME and agriculture expand — a systemic reallocation with major risk implications.</AlertBox>
      <div style={{ height: 16 }} />

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(218px,1fr))", gap: 16, marginBottom: 24 }}>
        {segmentData.map(s => {
          const irChange = s.ir25 - s.ir24;
          const improved = irChange < 0;
          return (
            <Card key={s.segment} style={{ border: `1px solid ${s.color}30` }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: s.color, fontFamily: "'Poppins',sans-serif", marginBottom: 13 }}>{s.segment}</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 9, marginBottom: 13 }}>
                <div style={{ textAlign: "center", background: `${s.color}0d`, borderRadius: 7, padding: 10 }}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: s.advGrowth < 0 ? A.red : A.green, fontFamily: "'Poppins',sans-serif" }}>{s.advGrowth > 0 ? "+" : ""}{s.advGrowth}%</div>
                  <div style={{ fontSize: 9.5, color: A.textLight, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>Advance Growth</div>
                </div>
                <div style={{ textAlign: "center", background: `${s.color}0d`, borderRadius: 7, padding: 10 }}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: s.nplChange > 0 ? A.red : A.green, fontFamily: "'Poppins',sans-serif" }}>{s.nplChange > 0 ? "+" : ""}{s.nplChange}%</div>
                  <div style={{ fontSize: 9.5, color: A.textLight, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>NPL Change</div>
                </div>
              </div>
              <div style={{ display: "flex", justifyContent: "space-around" }}>
                <Gauge value={s.ir24} max={22} label="Dec-24" color={A.midGray} size={72} />
                <Gauge value={s.ir25} max={22} label="Dec-25" color={improved ? A.green : A.red} size={72} />
              </div>
              {s.segment === "Agriculture" && <AlertBox type="critical">NPL ratio 9.8%→15.3%. NPLs surged +80.8%. Now the highest-risk segment in Pakistan's banking system.</AlertBox>}
              {s.segment === "Corporate" && <AlertBox type="info">Contracting (−11.4%). Capital flowing to SME. Risk repriced from large-corp to SME — intentional or not.</AlertBox>}
            </Card>
          );
        })}
      </div>

      <Card style={{ overflowX: "auto", marginBottom: 22 }}>
        <CardTitle>Table 8 — Segment-wise Advances &amp; NPLs (PKR millions)</CardTitle>
        <div style={{ height: 12 }} />
        <table>
          <thead>
            <tr>{["Segment","Adv Dec-24","Adv Dec-25","Adv Growth","NPL Dec-24","NPL Dec-25","NPL Change","IR Dec-24","IR Dec-25","Dir."].map((h,i) => <TH key={h} left={i===0}>{h}</TH>)}</tr>
          </thead>
          <tbody>
            {segmentData.map((s, i) => {
              const irUp = s.ir25 > s.ir24;
              return (
                <tr key={i} style={{ background: i % 2 === 0 ? A.surfaceWarm : A.surface, borderBottom: `1px solid ${A.border}` }}>
                  <td style={{ padding: "8px 11px", fontWeight: 600, color: s.color, fontFamily: "'Poppins',sans-serif" }}>{s.segment}</td>
                  <td style={{ padding: "8px 11px", textAlign: "right", fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{(s.adv24/1000).toFixed(0)}</td>
                  <td style={{ padding: "8px 11px", textAlign: "right", fontWeight: 600, fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{(s.adv25/1000).toFixed(0)}</td>
                  <td style={{ padding: "8px 11px", textAlign: "center", color: s.advGrowth < 0 ? A.red : A.green, fontWeight: 600, fontFamily: "'Poppins',sans-serif" }}>{s.advGrowth > 0 ? "+" : ""}{s.advGrowth}%</td>
                  <td style={{ padding: "8px 11px", textAlign: "right", fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{(s.npl24/1000).toFixed(0)}</td>
                  <td style={{ padding: "8px 11px", textAlign: "right", fontWeight: 600, fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{(s.npl25/1000).toFixed(0)}</td>
                  <td style={{ padding: "8px 11px", textAlign: "center", color: s.nplChange > 0 ? A.red : A.green, fontWeight: 600, fontFamily: "'Poppins',sans-serif" }}>{s.nplChange > 0 ? "+" : ""}{s.nplChange}%</td>
                  <td style={{ padding: "8px 11px", textAlign: "center", fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{s.ir24}%</td>
                  <td style={{ padding: "8px 11px", textAlign: "center", fontWeight: 700, color: irUp ? A.red : A.green, fontFamily: "'Poppins',sans-serif" }}>{s.ir25}%</td>
                  <td style={{ padding: "8px 11px", textAlign: "center", fontSize: 15, color: irUp ? A.red : A.green }}>{irUp ? "↑" : "↓"}</td>
                </tr>
              );
            })}
            <tr style={{ background: A.cerulean }}>
              {["Total System","16,914","15,896","-5.9%","1,068","964","-9.7%","6.3%","6.1%","↓"].map((v, ci) => (
                <td key={ci} style={{ padding: "9px 11px", color: ci === 3 ? "#f9c9c9" : ci === 6 ? "#c9f0c9" : ci === 8 ? "#c9f0c9" : A.light, fontWeight: ci === 0 || ci === 2 || ci === 8 ? 700 : 400, textAlign: ci === 0 ? "left" : "right", fontFamily: ci === 0 ? "'Poppins',sans-serif" : "'Poppins',sans-serif", fontSize: 12 }}>{v}</td>
              ))}
            </tr>
          </tbody>
        </table>
        <AlertBox type="critical" title="Total system NPL improving — but composition worsening">System IR improved 6.3% → 6.1%. The improvement is entirely from corporate NPL recovery. SME and Agriculture risk is RISING. System headline masks segment-level deterioration.</AlertBox>
      </Card>
    </div>
  );
}

// ─── RISK TAB ──────────────────────────────────────────────────────────────────
function RiskTab({ activeElIdx, setActiveElIdx }) {
  return (
    <div>
      <SectionHeader number="5" title="Updated Credit Risk & Expected Loss Quantification" subtitle="Updated PD/LGD assumptions incorporating Dec-25 data and unseasoned vintage risk" />
      <AlertBox type="new" title="EL Estimates Significantly Revised Upward">Dec-25 data has materially changed the risk picture. Base case EL: PKR 53.7 Bn (up from PKR 10.3 Bn). Blended LGD rises to 63% as new-vintage exposure now = 39.9% of total EAD.</AlertBox>
      <div style={{ height: 14 }} />

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 22 }}>
        <Card>
          <CardTitle>Updated PD &amp; LGD Assumptions</CardTitle>
          <div style={{ marginBottom: 18, marginTop: 10 }}>
            <div style={{ fontSize: 12.5, fontWeight: 600, color: A.dark, fontFamily: "'Poppins',sans-serif", marginBottom: 9, letterSpacing: "0.02em" }}>Probability of Default (PD)</div>
            {[["Optimistic","7.2%",A.green,"Improving quality; benign macro"],["Base","9.0%",A.blue,"Conservative: unseasoned vintage risk"],["Stress","10.8%",A.red,"Macro shock + higher NTB defaults"]].map(([sc,val,col,note])=>(
              <div key={sc} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8, padding: "8px 12px", background: `${col}0c`, borderRadius: 7, border: `1px solid ${col}20` }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: col, minWidth: 68, fontFamily: "'Poppins',sans-serif" }}>{sc}</span>
                <span style={{ fontSize: 19, fontWeight: 700, color: col, fontFamily: "'Poppins',sans-serif", minWidth: 52, letterSpacing: "-0.01em" }}>{val}</span>
                <span style={{ fontSize: 11, color: A.textMid, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>{note}</span>
              </div>
            ))}
          </div>
          <div style={{ marginBottom: 14 }}>
            <div style={{ fontSize: 12.5, fontWeight: 600, color: A.dark, fontFamily: "'Poppins',sans-serif", marginBottom: 9, letterSpacing: "0.02em" }}>Loss Given Default (LGD)</div>
            {[["Legacy book","55%",A.green,"Collateral & recovery history"],["New vintage (39.9%)","75%",A.red,"Unsecured / small ticket; NTB"],["Blended Base","63.0%",A.blue,"Weighted: 60.1% legacy × 39.9% new"]].map(([cat,val,col,note])=>(
              <div key={cat} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8, padding: "8px 12px", background: `${col}0c`, borderRadius: 7, border: `1px solid ${col}20` }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: col, minWidth: 104, fontFamily: "'Poppins',sans-serif" }}>{cat}</span>
                <span style={{ fontSize: 17, fontWeight: 700, color: col, fontFamily: "'Poppins',sans-serif", minWidth: 50 }}>{val}</span>
                <span style={{ fontSize: 11, color: A.textMid, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>{note}</span>
              </div>
            ))}
          </div>
          <AlertBox type="warning">New-vintage exposures = 39.9% of EAD (net growth Jun-24→Dec-25 ÷ Dec-25 total). These have NO repayment history in a downturn. LGD of 75% may still be optimistic for ISIC-95 and agriculture.</AlertBox>
        </Card>

        <Card>
          <CardTitle>Table 9 — Updated Expected Loss Scenarios</CardTitle>
          <CardSub>EAD: PKR 947.9 Bn · EL = EAD × PD × LGD · Click to see interpretation</CardSub>
          {elScenarios.map((s, i) => (
            <div key={i} onClick={() => setActiveElIdx(i)} style={{
              padding: "14px 16px", borderRadius: 9, marginBottom: 10, cursor: "pointer",
              background: activeElIdx === i ? `${s.color}0d` : A.surfaceWarm,
              border: `1.5px solid ${activeElIdx === i ? s.color : A.border}`,
              transition: "all 0.18s"
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                <span style={{ fontSize: 13, fontWeight: 600, color: A.dark, fontFamily: "'Poppins',sans-serif" }}>{s.scenario}</span>
                <span style={{ fontSize: 24, fontWeight: 700, color: s.color, fontFamily: "'Poppins',sans-serif", letterSpacing: "-0.02em" }}>PKR {s.el} Bn</span>
              </div>
              <div style={{ display: "flex", gap: 16, marginBottom: 8 }}>
                <span style={{ fontSize: 11, color: A.textLight, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>PD: {s.pd}%</span>
                <span style={{ fontSize: 11, color: A.textLight, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>LGD: {s.lgd}%</span>
                <span style={{ fontSize: 11, fontWeight: 600, color: s.color, fontFamily: "'Poppins',sans-serif" }}>EL: {((s.el/s.ead)*100).toFixed(1)}% of EAD</span>
              </div>
              <div style={{ height: 5, background: A.lightGray, borderRadius: 3 }}>
                <div style={{ height: "100%", width: `${(s.el / 80) * 100}%`, background: s.color, borderRadius: 3 }} />
              </div>
              {activeElIdx === i && <div style={{ fontSize: 12, color: A.text, marginTop: 10, padding: "8px 12px", background: `${A.surface}`, borderRadius: 6, fontFamily: "'Lora','Georgia',serif", border: `1px solid ${A.border}` }}>{s.note}</div>}
            </div>
          ))}
        </Card>
      </div>

      <Card>
        <CardTitle>Data Integrity Flags — Required Before Using Any Exposure Estimate</CardTitle>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(275px,1fr))", gap: 12, marginTop: 12 }}>
          <AlertBox type="critical" title="Three-Table Definition Drift">Table 3.14 (ISIC-SMEs) ≠ Table 1.9 (Segment SMEs) ≠ Sep SME Snapshot. A reconciliation map is required before any number becomes official policy input.</AlertBox>
          <AlertBox type="warning" title="Dec-25 +10.5% MoM Jump">+PKR 90.4 Bn in one month is anomalous. If it reverses in Jan-26, the Dec-25 peak should not be used as the EAD baseline.</AlertBox>
          <AlertBox type="warning" title="Denominator Effect in NPL Ratios">SME infection ratio improved 18.0% → 12.5%. But NPLs declined only −7.5% while outstanding grew +33.2%. Vintage analysis is critical.</AlertBox>
          <AlertBox type="info" title="Priority Data Gaps">RCS origination data by SE/ME/sector, 12M default rates by vintage, restructured/evergreened exposures, sector-level NPL ratios, and unique borrower IDs.</AlertBox>
        </div>
      </Card>
    </div>
  );
}

// ─── FISCAL TAB ───────────────────────────────────────────────────────────────
function FiscalTab({ activeFiscalIdx, setActiveFiscalIdx }) {
  return (
    <div>
      <SectionHeader number="6" title="Updated RCS Fiscal Risk & Claim Forecast" subtitle="Table 10 · Net growth proxy EAD · First-loss exposure · 12-month expected claims" />

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 22 }}>
        <Card>
          <CardTitle>Table 2 — Proxy RCS-Eligible EAD</CardTitle>
          <CardSub>Net-growth proxy since Jun-24 · does not adjust for repayments or rollovers</CardSub>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 11, marginBottom: 16, marginTop: 4 }}>
            {[["Jun-24 Baseline","PKR 570.1 Bn",A.midGray],["Dec-25 Latest","PKR 947.9 Bn",A.cerulean],["Net Growth (EAD)","PKR 377.8 Bn",A.green]].map(([l,v,c])=>(
              <div key={l} style={{ textAlign: "center", background: `${c}0e`, borderRadius: 9, padding: 14, border: `1px solid ${c}25` }}>
                <div style={{ fontSize: 16, fontWeight: 700, color: c, fontFamily: "'Poppins',sans-serif", lineHeight: 1.1, letterSpacing: "-0.01em" }}>{v}</div>
                <div style={{ fontSize: 10, color: A.textLight, marginTop: 5, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>{l}</div>
              </div>
            ))}
          </div>
          <div style={{ background: "#eef3f7", borderRadius: 8, padding: 14, marginBottom: 14, border: `1px solid ${A.blue}25` }}>
            <div style={{ fontSize: 12.5, fontWeight: 600, color: A.blue, fontFamily: "'Poppins',sans-serif", marginBottom: 7 }}>RCS Program Mix Assumption</div>
            {[["45% Small Enterprises (SE)","20% first-loss cover",A.green],["55% Medium Enterprises (ME)","10% first-loss cover",A.blue]].map(([cat,cov,col])=>(
              <div key={cat} style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: 12, fontFamily: "'Lora','Georgia',serif" }}>
                <span style={{ color: A.text }}>{cat}</span>
                <span style={{ fontWeight: 500, color: col, fontFamily: "'Poppins',sans-serif" }}>{cov}</span>
              </div>
            ))}
            <div style={{ borderTop: `1px solid ${A.border}`, marginTop: 8, paddingTop: 8 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, fontWeight: 700, fontFamily: "'Poppins',sans-serif" }}>
                <span style={{ color: A.cerulean }}>Weighted avg first-loss:</span>
                <span style={{ color: A.cerulean }}>14.5%</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, fontWeight: 700, fontFamily: "'Poppins',sans-serif" }}>
                <span style={{ color: A.cerulean }}>First-loss exposure:</span>
                <span style={{ color: A.cerulean }}>PKR 54.8 Bn</span>
              </div>
            </div>
          </div>
          <AlertBox type="warning">PKR 54.8 Bn first-loss exposure. MoF must provision against this in MTDS 2026-28.</AlertBox>
        </Card>

        <Card>
          <CardTitle>Table 10 — Ministry Expected Claims (12-Month Horizon)</CardTitle>
          <CardSub>Click scenario · Indicative — replace with vintage-level data when available</CardSub>
          {fiscalData.map((s, i) => (
            <div key={i} onClick={() => setActiveFiscalIdx(i)} style={{
              padding: "15px 17px", borderRadius: 9, marginBottom: 10, cursor: "pointer",
              background: activeFiscalIdx === i ? `${s.color}0d` : A.surfaceWarm,
              border: `1.5px solid ${activeFiscalIdx === i ? s.color : A.border}`,
              transition: "all 0.18s"
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: A.dark, fontFamily: "'Poppins',sans-serif" }}>{s.scenario}</div>
                  <div style={{ fontSize: 11.5, color: A.textMid, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>PD: {s.pd}% · First-loss: PKR {s.firstLoss} Bn</div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 27, fontWeight: 700, color: s.color, fontFamily: "'Poppins',sans-serif", lineHeight: 1, letterSpacing: "-0.02em" }}>PKR {s.claims} Bn</div>
                  <div style={{ fontSize: 10, color: A.textLight, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>12-month claims</div>
                </div>
              </div>
              <div style={{ marginTop: 9, height: 5, background: A.lightGray, borderRadius: 3 }}>
                <div style={{ height: "100%", width: `${(s.claims / 8) * 100}%`, background: s.color, borderRadius: 3 }} />
              </div>
            </div>
          ))}
          <AlertBox type="info">Expected claims are indicative and should be replaced with vintage-level observed default rates once available. Current estimates use proxy EAD — not actual origination data.</AlertBox>
        </Card>
      </div>

      <Card style={{ overflowX: "auto" }}>
        <CardTitle>How the Fiscal Picture Changed: Jun-25 vs Dec-25 Update</CardTitle>
        <div style={{ height: 12 }} />
        <table>
          <thead>
            <tr>{["Parameter","Original (Jun-25)","Updated (Dec-25)","Change"].map((h,i) => <TH key={h} left={i===0||i===3}>{h}</TH>)}</tr>
          </thead>
          <tbody>
            {[
              ["Total SME Outstanding","PKR 761.6 Bn","PKR 947.9 Bn","↑ +PKR 186.3 Bn (+24.5%)"],
              ["Proxy RCS EAD (net growth)","~PKR 187.0 Bn","PKR 377.8 Bn","↑ +PKR 190.8 Bn (+102%)"],
              ["First-loss exposure","~PKR 27.1 Bn","PKR 54.8 Bn","↑ +PKR 27.7 Bn (+102%)"],
              ["Base PD assumption","~2.0%","9.0%","↑ Methodology change"],
              ["Blended LGD","~55–60%","63.0%","↑ New vintage 39.9%"],
              ["Base EL estimate","PKR 10.3 Bn","PKR 53.7 Bn","↑↑ +PKR 43.4 Bn"],
              ["Central claims (12M)","~PKR 3–5 Bn","PKR 4.7 Bn","Consistent range"],
              ["Stress claims (12M)","~PKR 8–14 Bn","PKR 7.4 Bn","Slightly lower (1-year horizon)"],
            ].map((r, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? A.surfaceWarm : A.surface, borderBottom: `1px solid ${A.border}` }}>
                <td style={{ padding: "8px 11px", fontFamily: "'Lora','Georgia',serif", fontSize: 12 }}>{r[0]}</td>
                <td style={{ padding: "8px 11px", textAlign: "center", color: A.textMid, fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{r[1]}</td>
                <td style={{ padding: "8px 11px", textAlign: "center", fontWeight: 600, color: A.cerulean, fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{r[2]}</td>
                <td style={{ padding: "8px 11px", fontWeight: 500, color: r[3].includes("↑↑") ? A.red : r[3].includes("↑") ? A.orange : A.green, fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{r[3]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <AlertBox type="critical" title="Base EL nearly 5× higher in Dec-25 framework">Methodology shift + 102% growth in proxy EAD drives base EL from PKR 10.3 Bn to PKR 53.7 Bn. Both frameworks are internally consistent — the Dec-25 figure is more conservative and appropriate given unseasoned vintage risk.</AlertBox>
      </Card>
    </div>
  );
}

// ─── ALERTS TAB ───────────────────────────────────────────────────────────────
function AlertsTab() {
  const signals = [
    { title: "Agriculture NPL Surge", icon: "🌾", value: "15.3%", sub: "from 9.8% Dec-24 · +80.8% abs.", body: "Agriculture infection ratio is now the HIGHEST among all major banking segments. NPLs rose +80.8% in absolute terms even as system NPLs fell 9.7%. Requires immediate granular breakdown by crop, geography, and value chain node.", alert: "critical", color: A.red },
    { title: "Dec-25 Data Spike — Validate", icon: "📊", value: "+10.5%", sub: "MoM Dec-25 · +PKR 90.4 Bn", body: "The December surge is the largest single-month jump in the dataset. 63% WC-driven, 85% from 3 sectors. If Jan-26 reverses, this should not be the EAD baseline for fiscal planning.", alert: "warning", color: A.orange },
    { title: "Public Bank Underwriting", icon: "🏦", value: "+71.3%", sub: "public sector bank SME growth", body: "Share rose 26.9% → 32.1%. Historically the weakest underwriting channel. Program-driven targets suspected. Thematic inspection mandatory for banks with >150% YoY SME growth.", alert: "critical", color: A.red },
    { title: "Services SME Concentration", icon: "⚙", value: "+71.4%", sub: "Services SME YoY · 32.4% share", body: "Services SMEs the fastest-growing type and now 32.4% of the book. Asset-light, cash-volatile, often informal. Combined with agriculture NPL surge — two weakest collateral segments growing fastest.", alert: "warning", color: A.orange },
    { title: "Fixed Investment Cliff", icon: "🏗", value: "49.6%", sub: "of SME book in Fixed Investment", body: "FI share rose from 43.3% to 49.6%. FI loans from 2022-23 now at their 3-year refinancing trigger at sustained high KIBOR — stealth NPL risk not visible in current ratios.", alert: "warning", color: A.orange },
    { title: "Ticket Compression Paradox", icon: "📉", value: "−13.6%", sub: "avg ticket Sep-24→Sep-25", body: "Avg outstanding per borrower fell PKR 2.69M → 2.32M. More borrowers, smaller tickets = higher admin cost per PKR, higher default frequency, and lower recovery per default event.", alert: "info", color: A.blue },
  ];

  return (
    <div>
      <SectionHeader number="7" title="Early Warning System — Combined Risk Signals" subtitle="Alert Level: AMBER · Updated with Dec-25 data · SBP thematic inspection priorities" />

      {/* Amber banner */}
      <div style={{ background: A.orange, borderRadius: 10, padding: "17px 22px", marginBottom: 24, display: "flex", alignItems: "center", gap: 18 }}>
        <div style={{ fontSize: 34, color: A.light }} className="pulse">⚠</div>
        <div>
          <div style={{ fontSize: 17, fontWeight: 700, color: A.light, fontFamily: "'Poppins',sans-serif" }}>AMBER — HIGH VIGILANCE</div>
          <div style={{ fontSize: 12, color: "rgba(250,249,245,0.82)", fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>Three simultaneous risk signals: Agriculture NPL surge · Dec-25 data spike · Ticket compression — Banking Supervision Department, Feb 2026</div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(275px,1fr))", gap: 14, marginBottom: 22 }}>
        {signals.map(e => (
          <Card key={e.title} style={{ border: `1px solid ${e.color}30` }}>
            <div style={{ fontSize: 20, marginBottom: 8 }}>{e.icon}</div>
            <div style={{ fontSize: 13, fontWeight: 600, color: e.color, fontFamily: "'Poppins',sans-serif", marginBottom: 4 }}>{e.title}</div>
            <div style={{ fontSize: 25, fontWeight: 700, color: e.color, fontFamily: "'Poppins',sans-serif", lineHeight: 1, letterSpacing: "-0.02em" }}>{e.value}</div>
            <div style={{ fontSize: 10.5, color: A.textLight, marginBottom: 10, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>{e.sub}</div>
            <div style={{ fontSize: 12, color: A.text, lineHeight: 1.65, fontFamily: "'Lora','Georgia',serif" }}>{e.body}</div>
          </Card>
        ))}
      </div>

      <Card>
        <CardTitle>Data Quality Signals — Priority Reconciliation Items</CardTitle>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(275px,1fr))", gap: 10, marginTop: 12 }}>
          <AlertBox type="critical" title="Three-Dataset Discrepancy">Table 3.14: PKR 947.9 Bn Dec-25. Table 1.9: PKR 902.9 Bn Dec-25. Sep Snapshot: PKR 686.2 Bn Sep-25. Different purposes — but using wrong dataset for wrong purpose creates material policy error.</AlertBox>
          <AlertBox type="warning" title="Borrower Double-Counting Risk">Without unique borrower IDs across bank submissions, the 295,291 count may include double-counts. True unique borrower count likely lower.</AlertBox>
          <AlertBox type="warning" title="Professional & Scientific Decline">Sector L declined −13.9% since Jun-24 — only sector with negative contribution. Three possible causes: reclassification, de-risking, or demand contraction. Must distinguish before drawing conclusions.</AlertBox>
          <AlertBox type="info" title="Restructured/Evergreened Exposures">Net outstanding changes cannot separate new credit from rollovers. Without flow data, hidden stress is likely masked during the rapid growth phase.</AlertBox>
        </div>
      </Card>
    </div>
  );
}

// ─── ANNEX A ──────────────────────────────────────────────────────────────────
function AnnexTab() {
  const totalRow = annexAData.reduce((acc, r) => ({
    total: acc.total + r.total, tfin: acc.tfin + r.tfin, wc: acc.wc + r.wc,
    fi: acc.fi + r.fi, const_: acc.const_ + r.const_, other: acc.other + r.other
  }), { total: 0, tfin: 0, wc: 0, fi: 0, const_: 0, other: 0 });

  return (
    <div>
      <SectionHeader number="A" title="Annex A — Full Sector Totals by Financing Type (Dec-25)" subtitle="ISIC Table 3.14 · PKR Millions · Trade Finance / Working Capital / Fixed Investment / Construction / Other" />
      <AlertBox type="new" title="Complete Type Breakdown — New Data">Full Dec-25 sector × financing-type matrix. Enables sector-level WC rollover risk analysis, FI refinancing risk, and collateral assessment by sector.</AlertBox>
      <div style={{ height: 14 }} />

      <Card style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr>{["Sector","Total (PKR Mn)","Trade Finance","Working Capital","Fixed Investment","Construction","Other"].map((h,i) => <TH key={h} left={i===0}>{h}</TH>)}</tr>
            <tr style={{ background: A.cyan }}>
              {["","100%","% of total","% of total","% of total","% of total","% of total"].map((h, i) => (
                <th key={i} style={{ color: "rgba(250,249,245,0.65)", padding: "4px 11px", textAlign: i === 0 ? "left" : "center", fontFamily: "'Lora','Georgia',serif", fontStyle: "italic", fontSize: 9.5, fontWeight: 400 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {annexAData.map((r, i) => {
              const wcPct = ((r.wc / r.total) * 100).toFixed(0);
              const fiPct = ((r.fi / r.total) * 100).toFixed(0);
              return (
                <tr key={i} style={{ background: i % 2 === 0 ? A.surfaceWarm : A.surface, borderBottom: `1px solid ${A.border}` }}>
                  <td style={{ padding: "7px 11px", fontFamily: "'Lora','Georgia',serif", fontSize: 11.5, color: A.text }}>{r.sector}</td>
                  <td style={{ padding: "7px 11px", textAlign: "right", fontWeight: 700, color: A.cerulean, fontFamily: "'Poppins',sans-serif", fontSize: 12 }}>{r.total.toLocaleString()}</td>
                  <td style={{ padding: "7px 11px", textAlign: "right", color: A.textLight, fontFamily: "'Poppins',sans-serif", fontSize: 11 }}>{r.tfin > 0 ? r.tfin.toLocaleString() : "—"}</td>
                  <td style={{ padding: "7px 11px", textAlign: "right", color: parseInt(wcPct) > 50 ? A.cerulean : A.textLight, fontWeight: parseInt(wcPct) > 50 ? 600 : 400, fontFamily: "'Poppins',sans-serif", fontSize: 11 }}>
                    {r.wc.toLocaleString()} <span style={{ fontSize: 9.5, color: A.textLight }}>({wcPct}%)</span>
                  </td>
                  <td style={{ padding: "7px 11px", textAlign: "right", color: parseInt(fiPct) > 50 ? A.orange : A.textLight, fontWeight: parseInt(fiPct) > 50 ? 600 : 400, fontFamily: "'Poppins',sans-serif", fontSize: 11 }}>
                    {r.fi.toLocaleString()} <span style={{ fontSize: 9.5, color: A.textLight }}>({fiPct}%)</span>
                  </td>
                  <td style={{ padding: "7px 11px", textAlign: "right", color: A.textLight, fontFamily: "'Poppins',sans-serif", fontSize: 11 }}>{r.const_ > 0 ? r.const_.toLocaleString() : "—"}</td>
                  <td style={{ padding: "7px 11px", textAlign: "right", color: A.textLight, fontFamily: "'Poppins',sans-serif", fontSize: 11 }}>{r.other > 0 ? r.other.toLocaleString() : "—"}</td>
                </tr>
              );
            })}
            <tr style={{ background: A.cerulean }}>
              <td style={{ padding: "9px 11px", color: A.light, fontWeight: 700, fontFamily: "'Poppins',sans-serif" }}>TOTAL</td>
              {[totalRow.total, totalRow.tfin, totalRow.wc, totalRow.fi, totalRow.const_, totalRow.other].map((v, i) => (
                <td key={i} style={{ padding: "9px 11px", textAlign: "right", color: A.light, fontWeight: 700, fontFamily: "'Poppins',sans-serif" }}>{Math.round(v).toLocaleString()}</td>
              ))}
            </tr>
          </tbody>
        </table>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(195px,1fr))", gap: 13, marginTop: 16 }}>
        {[
          { label: "WC-Dominant Sectors", val: "Manufacturing, Trade", note: "Rollover risk · Liquidity cliff if banks tighten", color: A.cerulean },
          { label: "FI-Dominant Sectors", val: "Transport, Education, Real Estate", note: "Refinancing risk at sustained high rates", color: A.orange },
          { label: "Balanced Mix Sectors", val: "Construction, Services, ICT", note: "Mixed collateral profile · Moderate risk", color: A.blue },
          { label: "Trade Finance Leaders", val: "Manufacturing (PKR 29.4 Bn), Trade (PKR 5.2 Bn)", note: "FX and documentary risk exposure", color: A.green },
        ].map(c => (
          <Card key={c.label} style={{ borderLeft: `3px solid ${c.color}` }}>
            <div style={{ fontSize: 10.5, fontWeight: 600, color: c.color, fontFamily: "'Poppins',sans-serif", marginBottom: 4 }}>{c.label}</div>
            <div style={{ fontSize: 12.5, fontFamily: "'Lora','Georgia',serif", color: A.text }}>{c.val}</div>
            <div style={{ fontSize: 10.5, color: A.textLight, marginTop: 4, fontFamily: "'Lora','Georgia',serif", fontStyle: "italic" }}>{c.note}</div>
          </Card>
        ))}
      </div>
    </div>
  );
}

// ─── RECOMMENDATIONS TAB ──────────────────────────────────────────────────────
function RecommendationsTab({ activeRec, setActiveRec }) {
  const stakeholders = [
    { id: "ncgcl", label: "NCGCL",               color: A.cerulean },
    { id: "mof",   label: "Ministry of Finance", color: A.orange   },
    { id: "sbp",   label: "SBP / Regulators",    color: A.green    },
  ];

  const recs = {
    ncgcl: [
      { p: "CRITICAL", action: "Quarterly Vintage Dashboard", detail: "Implement quarterly tracking of RCS guarantee claims by origination quarter × default rate × SE/ME × secured/unsecured. The single most important analytical investment — without it, no fiscal forecast is reliable. Build within 60 days." },
      { p: "CRITICAL", action: "Sub-sector Concentration Caps", detail: "Cap ISIC Q (Other Services) at 5% of guaranteed portfolio. Agriculture (sector A) at 12% with mandatory disaggregation by crop type and geography. ISIC J (ICT): 1.5% with LGD floor of 80% for FI loans given technology obsolescence." },
      { p: "CRITICAL", action: "Agriculture Guarantee Redesign", detail: "Align agriculture guarantee coverage with verified cashflow collateral — warehouse receipts, confirmed buyer contracts, crop insurance policies. Plain-vanilla guarantees on crop production loans with 15.3% NPL infection are no longer appropriate." },
      { p: "HIGH", action: "SME Type-Differentiated Products", detail: "Price separately for Services SMEs (higher EL, lower LGD) vs Manufacturing SMEs (lower EL, tangible collateral). Services SMEs at current pricing are cross-subsidized by manufacturing — distorting credit allocation." },
      { p: "HIGH", action: "Ticket Size Repricing", detail: "Re-price guarantees to reflect expected loss + admin cost for sub-PKR 2.5M facilities. Current pricing likely subsidizes the NTB segment at the expense of larger, safer borrowers." },
      { p: "HIGH", action: "Islamic SME Framework", detail: "PKR 83 Bn in Islamic bank SME credit requires dedicated monitoring. Murabaha and Ijara structures have different collateral crystallization mechanics — standard NPL trigger frameworks may miss early distress." },
    ],
    mof: [
      { p: "CRITICAL", action: "Recognize PKR 54.8 Bn First-Loss in MTDS", detail: "MTDS 2026-28 must explicitly show PKR 54.8 Bn as a contingent liability. Under stress (PD 13.5%), 12-month claims = PKR 7.4 Bn. Budget Wave 1 and Wave 2 as distinct line items — not smoothed annual provisions." },
      { p: "CRITICAL", action: "Validate Dec-25 Spike Before Budgeting", detail: "The Dec-25 +PKR 90.4 Bn MoM jump MUST be validated before using Dec-25 as the EAD baseline. If seasonal/technical, true EAD and fiscal liability is materially lower than PKR 377.8 Bn." },
      { p: "CRITICAL", action: "Agriculture Separate Provision", detail: "Agriculture NPL infection ratio reached 15.3% — the highest in the system. Commission a standalone agriculture credit risk assessment for budget FY 2026-27." },
      { p: "HIGH", action: "Fiscal-Monetary Coordination Protocol", detail: "MPC rate hike decisions must formally factor the SME Fiscal Trigger. +200 bps adds PKR 18.7 Bn interest burden to SMEs. +400 bps exhausts MFB guarantee cushions and risks credit crunch reversing inclusion gains." },
      { p: "HIGH", action: "Guarantee Restructure: Reduce Moral Hazard", detail: "Transition from 20% flat first-loss to '10% first-loss + 30% pari-passu' for SE facilities. Banks must co-share risk above the first tranche. Most cost-effective moral hazard mitigation without withdrawing the guarantee." },
      { p: "MEDIUM", action: "Data Reconciliation Mandate", detail: "Commission SBP to produce a formal reconciliation map across Table 3.14, Table 1.9, and Sep SME Snapshot within 90 days. Without this, no exposure figure in budget documents can be audited or defended." },
    ],
    sbp: [
      { p: "CRITICAL", action: "Three-Table Reconciliation Map", detail: "Produce a formal reconciliation between Table 3.14, Table 1.9, and Sep SME Snapshot within 60 days. These are the three primary datasets feeding fiscal risk estimates — their discrepancy is a data integrity failure, not a rounding issue." },
      { p: "CRITICAL", action: "Dec-25 Spike Investigation", detail: "Determine whether the +10.5% MoM Dec-25 surge was seasonal WC drawdowns, reporting lags, classification changes, or genuine credit acceleration. Issue a data quality note and revise year-end reference period guidance." },
      { p: "CRITICAL", action: "Vintage Dashboard — Mandatory Reporting", detail: "Require all RCS banks to report: origination quarter, facility type, SE vs ME, secured vs unsecured, delinquency status (30/60/90-day), and restructured/evergreened flag. Monthly reporting with 15-day lag. Essential for Wave 2 forecast validation." },
      { p: "HIGH", action: "Thematic Inspection — Public Sector Banks", detail: "Public sector banks grew SME book +71.3% in 12 months. Conduct thematic inspection of SME underwriting standards, NTB borrower screening, and program compliance at NBP and BoP." },
      { p: "HIGH", action: "Cash-Flow Lending Mandate for Loans >PKR 5M", detail: "Require digital transaction data (POS / bank statements / FBR e-tax) as a mandatory element for any SME facility above PKR 5M. Eliminates pure 'clean' lending above this threshold." },
      { p: "HIGH", action: "Agriculture Sub-Sector Reporting", detail: "Require sector-level NPL ratios for Wholesale/Retail, Manufacturing, Agriculture, Services, Transport. Agriculture NPL at 15.3% is systemic — SBP currently cannot identify which crops or geographies are driving deterioration." },
    ],
  };

  const pc = { CRITICAL: A.red, HIGH: A.orange, MEDIUM: A.blue };
  const activeStk = stakeholders.find(s => s.id === activeRec);

  return (
    <div>
      <SectionHeader number="8" title="Strategic Recommendations" subtitle="Prioritized actions for NCGCL, Ministry of Finance, and SBP · Updated with Dec-25 insights" />

      <div style={{ background: "#eef3f7", borderRadius: 10, padding: 18, marginBottom: 22, border: `1px solid ${A.blue}30` }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: A.blue, fontFamily: "'Poppins',sans-serif", marginBottom: 6 }}>Governing Principle — Catalytic Finance</div>
        <div style={{ fontSize: 12.5, color: A.text, lineHeight: 1.75, fontFamily: "'Lora','Georgia',serif" }}>
          NCGCL's mandate is Catalytic Finance: using limited risk capital to unlock durable SME growth by pricing and sharing risk through structured products. The Dec-25 data shows the <em>volume objective has been achieved</em> (PKR 947.9 Bn, +66.3%). The <em>next mandate is quality</em>: unseasoned vintages, agriculture NPL surge, and data gaps must be addressed before the 2026-27 guarantee cycle begins.
        </div>
      </div>

      <div style={{ display: "flex", gap: 9, marginBottom: 22 }}>
        {stakeholders.map(s => (
          <button key={s.id} className="rec-btn" onClick={() => setActiveRec(s.id)} style={{
            background: activeRec === s.id ? s.color : A.surface,
            color: activeRec === s.id ? A.light : s.color,
            border: `1.5px solid ${s.color}`,
            borderRadius: 8, padding: "9px 20px",
            cursor: "pointer", fontSize: 11.5, fontWeight: 600, fontFamily: "'Poppins',sans-serif",
            transition: "all 0.18s"
          }}>{s.label}</button>
        ))}
      </div>

      <div className="fi" key={activeRec}>
        {recs[activeRec].map((r, i) => (
          <div key={i} style={{
            background: A.surface,
            borderRadius: 10, padding: 20, marginBottom: 10,
            border: `1px solid ${A.border}`,
            borderLeft: `3px solid ${pc[r.p]}`,
            display: "flex", gap: 14, alignItems: "flex-start",
            boxShadow: "0 1px 5px rgba(20,20,19,0.05)"
          }}>
            <div style={{ flexShrink: 0, width: 64 }}>
              <div style={{
                background: `${pc[r.p]}14`, color: pc[r.p],
                padding: "3px 5px", borderRadius: 5,
                fontSize: 8.5, fontWeight: 700, fontFamily: "'Poppins',sans-serif",
                textAlign: "center", letterSpacing: "0.05em", border: `1px solid ${pc[r.p]}30`
              }}>{r.p}</div>
              <div style={{ textAlign: "center", fontSize: 20, fontWeight: 700, color: `${pc[r.p]}35`, fontFamily: "'Poppins',sans-serif", marginTop: 5 }}>{i + 1}</div>
            </div>
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, color: activeStk.color, fontFamily: "'Poppins',sans-serif", marginBottom: 5 }}>{r.action}</div>
              <div style={{ fontSize: 12.5, color: A.text, lineHeight: 1.72, fontFamily: "'Lora','Georgia',serif" }}>{r.detail}</div>
            </div>
          </div>
        ))}
      </div>

      <Card style={{ marginTop: 22 }}>
        <CardTitle>Priority Data Gaps — Required for Next Update</CardTitle>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(245px,1fr))", gap: 10, marginTop: 12 }}>
          {[
            { title: "RCS Origination Data", body: "Quarterly disbursements and outstanding by SE/ME, bank, sector (ISIC), and collateralization status. Without this, proxy EAD cannot be verified.", urgency: "CRITICAL" },
            { title: "Default & Recovery Data", body: "12-month and 24-month cumulative default rates by vintage. Recovery timelines to empirically calibrate PD/LGD. Currently inferred from infection ratios only.", urgency: "CRITICAL" },
            { title: "Restructured Exposures", body: "Counts and volumes of restructured/evergreened facilities. Rapid growth phases mask hidden stress through loan modifications.", urgency: "HIGH" },
            { title: "Borrower Quality Indicators", body: "NTB share by bank, repeat borrower count, geographic distribution, top-obligor concentration. Enable real exposure deduplication.", urgency: "HIGH" },
            { title: "Sector-Level NPL Ratios", body: "NPL ratios by sector (Wholesale/Retail, Manufacturing, Agriculture, Services) — not just overall SME. Currently unavailable from public SBP data.", urgency: "HIGH" },
            { title: "RCS SAAF Scheme Vintage", body: "SAAF-specific disbursement by month since Aug-2024, split by borrower type and collateral status. Clean (unsecured) facilities must be tracked separately.", urgency: "MEDIUM" },
          ].map(d => {
            const uc = { CRITICAL: A.red, HIGH: A.orange, MEDIUM: A.blue }[d.urgency];
            const ubg = { CRITICAL: "#fdf0f0", HIGH: "#fef8f4", MEDIUM: "#eef3f7" }[d.urgency];
            return (
              <div key={d.title} style={{ padding: 13, background: ubg, borderRadius: 8, border: `1px solid ${uc}30` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 6 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: A.dark, fontFamily: "'Poppins',sans-serif" }}>{d.title}</span>
                  <span style={{ fontSize: 8.5, fontWeight: 700, color: uc, fontFamily: "'Poppins',sans-serif", padding: "2px 6px", background: "rgba(255,255,255,0.7)", borderRadius: 8, border: `1px solid ${uc}40` }}>{d.urgency}</span>
                </div>
                <div style={{ fontSize: 11.5, color: A.text, lineHeight: 1.6, fontFamily: "'Lora','Georgia',serif" }}>{d.body}</div>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}
