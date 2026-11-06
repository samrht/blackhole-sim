import { describe, it, expect } from "vitest";
import { compPart, dsPart, sliceBetween, withFlowDl, withTol, FLOW_ENTRIES } from "../src/test/flow-arms";
import limitWGSL from "../src/render/limit.wgsl?raw";
import limitDsWGSL from "../src/render/limit-ds.wgsl?raw";
import integratorSharedWGSL from "../src/render/integrator-shared.wgsl?raw";
import raytraceWGSL from "../src/render/raytrace.wgsl?raw";

describe("cutting a shader into parts", () => {
  it("takes the text from one marker up to another", () => { expect(sliceBetween("aXbYc", "X", "Y")).toBe("Xb"); });
  it("refuses a missing marker rather than producing a wrong shader", () => {
    expect(() => sliceBetween("abc", "X", "c")).toThrow(); expect(() => sliceBetween("abc", "a", "X")).toThrow();
  });
});

describe("the compensated integrator without its route", () => {
  const c = compPart(limitWGSL);
  it("has the compensated step and no entry point or uniform of its own", () => {
    expect(c).toContain("fn cStep("); expect(c).toContain("fn twoSum4(");
    expect(c).not.toContain("@compute"); expect(c).not.toContain("U."); expect(c).not.toContain("fn minoComp");
  });
  it("is always on", () => { expect(c).toContain("if (true)"); });
  it("can be switched off, for the control", () => { expect(compPart(limitWGSL, false)).toContain("if (false)"); });
});

describe("the double-single integrator without its route", () => {
  const d = dsPart(limitDsWGSL);
  it("has the step and arithmetic and no entry point, binding or uniform", () => {
    expect(d).toContain("fn dStepOnce("); expect(d).toContain("fn dTwoSum("); expect(d).toContain("fn dRay(");
    expect(d).not.toContain("@compute"); expect(d).not.toContain("@group"); expect(d).not.toContain("U.");
  });
});

describe("rewriting a constant", () => {
  it("changes the tolerance in the integrator's text", () => {
    const s = withTol(integratorSharedWGSL, 1e-7);
    expect(s).toContain("const MINO_TOL = 1e-7;"); expect(s).not.toContain("const MINO_TOL = 1e-5;");
  });
  it("changes the flow's quadrature step in the renderer's text", () => {
    const s = withFlowDl(raytraceWGSL, 0.25);
    expect(s).toContain("const FLOW_DL = 0.25;"); expect(s).not.toContain("const FLOW_DL = 0.125;");
  });
  it("refuses to rewrite a constant that is not there", () => {
    expect(() => withTol("no such constant", 1e-7)).toThrow(); expect(() => withFlowDl("no such constant", 0.25)).toThrow();
  });
});

describe("the entry points", () => {
  it("run the renderer's own flow loop", () => {
    for (const e of [FLOW_ENTRIES.comp, FLOW_ENTRIES.ds]) { expect(e).toContain("flowSeg("); expect(e).toContain("@compute"); expect(e).toContain("BM_NONE"); expect(e).toContain("HF_RMAX"); }
  });
});
