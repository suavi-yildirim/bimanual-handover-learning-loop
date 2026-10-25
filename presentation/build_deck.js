// Builds presentation/C2_Teach_the_Robot.pptx from the figures in presentation/figures.
// Run: NODE_PATH=<dir with pptxgenjs>/node_modules node presentation/build_deck.js [applyThemeModule]
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const FIG = (n) => path.join(__dirname, "figures", n + ".png");
const OUT = path.join(__dirname, "C2_Teach_the_Robot.pptx");
const VIDEO = path.join(ROOT, "sim", "out", "expert_test", "ep_000", "video.mp4");
const POSTER = process.env.POSTER || path.join(__dirname, "assets", "demo_poster.png");
const HERO = path.join(__dirname, "assets", "sim_robot_closeup.png");

// Siemens Healthineers colour code: Healthy Orange on black or white, Petrol as accent.
const THEME = {
  name: "Siemens Healthineers C2",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "000000", lt1: "FFFFFF", dk2: "1B1B1B", lt2: "ECECEC",
    accent1: "EC6602", accent2: "009999", accent3: "2A247E", accent4: "D52D2C", accent5: "449745",
    accent6: "F9D447", hlink: "009999", folHlink: "71202F",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in, same 16:9 as the figures
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Teach the Robot: a physical AI learning loop for a cathlab companion";
pres.subject = "Siemens Healthineers Robotics & AI Hackathon 2026, Challenge 2";
pres.author = "Challenge 2 team";
const C = pres.SchemeColor;
const FOOTER = "Challenge 2 · Teach the Robot · Physical AI learning loop for a cathlab companion";

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "C2 Title Dark",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 2.0, w: 6.4, h: 1.9, fontSize: 44, bold: true,
      color: C.accent1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.6, y: 4.05, w: 6.2, h: 1.1, fontSize: 20,
      color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "meta", type: "body", x: 0.6, y: 6.25, w: 6.6, h: 0.7, fontSize: 12,
      color: C.background2, valign: "bottom", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({ title: "C2 Figure", background: { color: C.background1 }, objects: [] });
pres.defineSlideMaster({
  title: "C2 Title Only",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.48, y: 0.3, w: 12.4, h: 0.75, fontSize: 32,
      bold: true, color: C.accent1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.48, y: 1.02, w: 12.4, h: 0.45, fontSize: 16,
      color: C.text2, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: FOOTER, options: { x: 0.48, y: 7.15, w: 8, h: 0.25, fontSize: 9, color: "5A5A5A", margin: 0 } } },
  ],
});
pres.defineSlideMaster({
  title: "C2 Closing Dark",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.7, w: 12.1, h: 1.1, fontSize: 44,
      bold: true, color: C.accent1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.6, y: 1.8, w: 12.1, h: 0.6, fontSize: 20,
      color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});

// ---------------------------------------------------------------- helpers
function figureSlide(section, fig, alt, notes) {
  const s = pres.addSlide({ masterName: "C2 Figure", sectionTitle: section });
  s.addImage({ path: FIG(fig), x: 0, y: 0, w: 13.333, h: 7.5, altText: alt, objectName: fig });
  s.addNotes(notes);
  return s;
}

// ---------------------------------------------------------------- slides
pres.addSection({ title: "Opening" });
{
  const s = pres.addSlide({ masterName: "C2 Title Dark", sectionTitle: "Opening" });
  s.addText("Teach the Robot", { placeholder: "title" });
  s.addText("A physical AI learning loop for a cathlab robot companion", { placeholder: "subtitle" });
  s.addText("Siemens Healthineers Robotics & AI Hackathon 2026 · Challenge 2\nErlangen, 24 October 2026",
    { placeholder: "meta" });
  s.addImage({ path: HERO, x: 7.4, y: 0, w: 5.933, h: 7.5, sizing: { type: "cover", w: 5.933, h: 7.5 },
    altText: "Simulated FR3 Duo dual-arm robot placing a box on a serving tray in the Isaac Sim hospital room",
    objectName: "Hero image" });
  s.addNotes("Open with the request a nurse would make: 'Hand me the box and place it on the serving tray.' " +
    "Everything in this talk is what has to happen behind that one sentence.");
}
figureSlide("Opening", "F00_one_image_story",
  "One-image summary: the cathlab problem, the robot companion receiving a nurse's request, and the five-stage learning loop with key numbers 84 demos = 1 motion, 0 of 28 v0 runs complete the task, 16 of 18 expert demos succeed.",
  "The whole story on one slide. Left: the problem in the cathlab. Middle: our robot companion, driven by a pi0.5 " +
  "vision-language-action policy, nurse in charge (autonomy level 2, no patient contact). Right: the learning loop " +
  "01 to 05, where failures in simulation become the next training data. Three numbers we will come back to: " +
  "84 demos contain one motion, 0 of 28 v0 runs complete the task, 16 of 18 generated expert demos succeed.");

pres.addSection({ title: "Problem" });
figureSlide("Problem", "F01_problem",
  "The problem: cathlab nurses are short of time, traditional robots need programming per task, a physical AI companion learns from demonstrations. Bottom: six frames of the first skill, the box handover.",
  "Cathlab nurses prepare patients and equipment and assist the cardiologist, under acute staff shortage. " +
  "Traditional automation needs an engineer per object and per scene. We want a robot that learns from " +
  "demonstrations, and a loop that creates, evaluates and deploys those skills. Our first skill end to end: " +
  "hand a box from the left gripper to the right and place it on the serving tray.");
figureSlide("Problem", "F02_nurse_robot_interaction",
  "Swimlane diagram of nurse, robot, safety supervisor and learning loop across six steps: request, plan, execute, monitor, hand over, learn.",
  "The nurse asks in plain language and stays in charge. A skill orchestrator picks the skill, the policy executes, " +
  "and a supervisor checks every frame: grasp missed, stalled, box tipped, dropped. It first retries on its own; " +
  "only when retries are used up does it ask the nurse. The nurse confirms with one tap, which becomes a label. " +
  "Every run is logged; failures become training data. New versions reach a real robot only after human sign-off.");

pres.addSection({ title: "Solution" });
figureSlide("Solution", "F03_pipeline_flowchart",
  "Flowchart of the five stages Ingest and Transform, Dataset Intelligence, Fine-Tune, Evaluate, Version and Serve, each with tools and key result, with a feedback loop and an Isaac Sim screenshot.",
  "Five stages, each with a tool we built and a number. 01: converters for LeRobot and our sim demos. " +
  "02: the dataset checker. 03: pi0.5 LoRA fine-tuning with openpi. 04: offline and closed-loop evaluation in " +
  "Isaac Sim with milestones and confidence intervals. 05: a policy server plus versioned splits and configs. " +
  "Isaac Sim closes the loop: replay test, expert demo generator, closed-loop evaluation.");
figureSlide("Solution", "F04_policy_inputs_outputs",
  "Policy inputs (three camera images, instruction text, 16 joint values) flow into the pi0.5 model; output is a 10-step action chunk for both arms, plotted over time. Inference 95 ms of a 333 ms budget.",
  "One query: three 224 by 224 camera images, the instruction, and 16 joint values. Out comes a plan of 10 steps " +
  "for both arms and grippers, executed at 30 Hz. Inference takes about 95 ms, under a third of the 333 ms the " +
  "plan lasts, so control stays real-time.");
figureSlide("Solution", "F05_task_storyboard",
  "Storyboard of eight simulation frames from start to placed on tray, with left and right gripper state traces over 69 seconds and the four scored milestones.",
  "The task step by step: approach, left grasp, lift and carry, right grasp, left release, right carries, place. " +
  "We score four milestones: left grasp with the box lifted, right grasp, handover, and placed within 30 mm.");

pres.addSection({ title: "Evidence" });
figureSlide("Evidence", "F06_dataset_intelligence",
  "Dataset intelligence: cube set 490 distinct motions of 490, handover sets 1 distinct motion; idle frame shares; list of what the handover data cannot teach and what we did about it.",
  "Before training, our checker reads every episode. The cube set has 490 distinct motions. The handover set has 84 " +
  "episodes but only one motion: the box never moves, only the lighting changes. No corrections, no failures, and a " +
  "2.5 s pause without a visual cue. Verdict: not ready for a policy that must generalise.");
figureSlide("Evidence", "F07_v0_failure_analysis",
  "v0 closed-loop results over 28 development runs: left grasp 10 of 28, right grasp 1, handover 1, placed 0, with confidence intervals; how each run ended; five root causes.",
  "Offline the provided checkpoint looks great: 0.39 degrees error. In closed loop, 0 of 28 runs complete the task. " +
  "These are development runs with mixed settings, so read them as a diagnosis, not a benchmark. Five root causes, " +
  "each checked on traces: covariate shift, a narrow close window, the gripper value as a phase cue, the silent " +
  "pause, and a weak grip with jerky plans. Lesson: fix the data, and measure over many runs.");
figureSlide("Evidence", "F08_v1_data_generation",
  "Expert demo generation: scatter of box offsets with 16 successes and 2 failures inside a plus or minus 15 mm range, grasp alignment errors all below 4 mm, and key numbers.",
  "So we fixed the data. A scripted expert in Isaac Sim replays the task with the box moved up to 15 mm and 5 degrees, " +
  "corrects the grasp to under 4 mm, and adds small noise to the motion while keeping the clean label. 16 of 18 " +
  "episodes succeed. Switching to real-time rendering cut an episode from 34 to 6 minutes. 14 train, 2 held out.");
figureSlide("Evidence", "F10_v0_vs_v1_results",
  "Placeholder chart for the controlled v0 versus v1 comparison per milestone, with the test protocol.",
  "Controlled comparison: raw policy only, new box positions, same render mode, N runs each, 95% intervals. " +
  "UPDATE before presenting: fill presentation/data/v0_v1_eval.json and rebuild figures and deck. If v1 does not " +
  "improve, say so and show v2: DAgger corrections on v1's own rollouts.");

pres.addSection({ title: "Demo" });
{
  const s = pres.addSlide({ masterName: "C2 Title Only", sectionTitle: "Demo" });
  s.addText("Demo: the full handover in Isaac Sim", { placeholder: "title" });
  s.addText("Scripted expert, our data generator: head camera, left wrist and right wrist, 69 s",
    { placeholder: "subtitle" });
  const poster = "data:image/png;base64," + fs.readFileSync(POSTER).toString("base64");
  s.addMedia({ type: "video", path: VIDEO, x: 0.48, y: 1.75, w: 12.37, h: 4.12, cover: poster,
    objectName: "Demo video" });
  s.addText("Shown: the scripted expert that generates the v1 training data, with the box moved. " +
    "It is the data generator, not the learned policy.", { x: 0.48, y: 6.1, w: 12.37, h: 0.6, fontSize: 14, color: C.text2,
    margin: 0, isTextBox: true, objectName: "Demo caption" });
  s.addNotes("Click to play (30 s is enough: stop after the handover). Be explicit that this is the scripted expert " +
    "that generates training data. If a successful v1 rollout exists by the freeze, swap in its video.mp4.");
}

pres.addSection({ title: "Outlook" });
figureSlide("Outlook", "F11_bigger_picture_roadmap",
  "Roadmap from now (hackathon) to next (weeks), then (months) and later (network), and a grid of cathlab skills the same loop can teach.",
  "Same loop, more skills. Next: DAgger, more objects already in the scene (scissors, pill bottle, glass plate), a " +
  "VLM success judge. Then: the real FR3 Duo, nurse labels from the tablet, a skill library and an orchestrator. " +
  "Later: share validated skills across robots and sites. The nurse decides what the robot learns next.");
{
  const s = pres.addSlide({ masterName: "C2 Closing Dark", sectionTitle: "Outlook" });
  s.addText("The learning loop is the product", { placeholder: "title" });
  s.addText("Raw demonstrations in; a served, evaluated, reproducible robot skill out", { placeholder: "subtitle" });
  const stats = [
    ["84 → 1", "handover demos hold one motion: we measured it before training", "FFFFFF"],
    ["0 / 28", "v0 runs complete the task in closed loop: five causes found", "D52D2C"],
    ["16 / 18", "generated expert demos succeed with the box moved", "009999"],
  ];
  stats.forEach(([big, small, col], i) => {
    const x = 0.6 + i * 4.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 3.0, w: 3.8, h: 2.6, rectRadius: 0.08,
      fill: { color: "1B1B1B" }, line: { color: "1B1B1B" }, objectName: "Stat card " + (i + 1) });
    s.addText(big, { x: x + 0.3, y: 3.25, w: 3.2, h: 1.0, fontSize: 48, bold: true, color: col, margin: 0,
      isTextBox: true, objectName: "Stat value " + (i + 1) });
    s.addText(small, { x: x + 0.3, y: 4.35, w: 3.2, h: 1.0, fontSize: 15, color: "D0D0D0", margin: 0,
      valign: "top", isTextBox: true, objectName: "Stat label " + (i + 1) });
  });
  s.addText("Thank you. Questions?", { x: 0.6, y: 6.1, w: 12.1, h: 0.7, fontSize: 24, bold: true, color: C.accent1,
    margin: 0, isTextBox: true, objectName: "Closing line" });
  s.addNotes("Close: we did not just train one skill. We built the loop that tells you whether a skill is ready, " +
    "why it fails, and how to fix the data. Then open for questions; backup slides follow.");
}

pres.addSection({ title: "Backup" });
figureSlide("Backup", "F12_safety_layers",
  "Concentric safety layers: nurse, supervisor, interlocks, limits, around the pi0.5 policy, with descriptions.",
  "Backup for 'is it safe?': the learned policy never acts alone. Limits, interlocks, a runtime supervisor and the " +
  "nurse sit between it and the world.");
figureSlide("Backup", "F13_operator_interface",
  "Concept mock-up of the nurse's tablet: three live camera views, progress by milestone, status, a large STOP button and done right / not right labels.",
  "Backup for 'what does the nurse do?': a concept tablet built on the live view and run logs we already produce. " +
  "Status values on the mock-up are illustrative.");
figureSlide("Backup", "F09_offline_accuracy_latency",
  "Offline arm error 6.79 degrees for base pi0.5 versus 0.39 for fine-tuned; gripper error 0.248 versus 0.006; latency per plan in 28 sim runs well under the 333 ms budget.",
  "Backup for 'is pi0.5 fast and accurate enough?': fine-tuning cuts arm error from 6.79 to 0.39 degrees offline, " +
  "and latency stays well under budget. But offline error is not task success; that is why stage 04 runs in sim.");

module.exports = { pres, THEME, OUT };

if (require.main === module) {
  (async () => {
    await pres.writeFile({ fileName: OUT });
    const themeMod = process.argv[2];
    if (themeMod) {
      const { applyTheme } = require(themeMod);
      await applyTheme(OUT, THEME);
    }
    console.log("wrote", OUT);
  })();
}
