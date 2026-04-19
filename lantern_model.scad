// Ottoman Hanging Lantern — 2-part snap-fit design, 3D printable
//
// PART 1 (body): 3 stepped tiers — print with bottom tier on bed (flat, small base)
//                Each tier has window cutouts. Solid walls. Snap rim on top.
// PART 2 (roof): Barrel vault with closed ends, flaps, and hanging loop
//                Print base-plate-down. Small span = no supports needed.
//
// HOW TO EXPORT STLs:
//   1. Set `part = "body"` → Design menu → Render (F6) → File → Export → STL
//   2. Set `part = "roof"` → Design menu → Render (F6) → File → Export → STL
//   3. Slice each STL with 0.2mm layer, 2-3 perimeters, 15-20% infill
//   4. No supports needed for either part

$fn = 60;

part = "both";   // "both", "body", or "roof"

// ─── Parameters ───

wall = 2.0;          // Wall thickness (5 perimeters at 0.4mm)

// Tier outer dimensions (width × depth × height)
top_w = 72;  top_d = 52;  top_h = 30;
mid_w = 50;  mid_d = 40;  mid_h = 22;
bot_w = 32;  bot_d = 26;  bot_h = 16;

// Tier cap thickness (solid top/bottom plates on each tier)
cap_t = 2.0;

// Gap between tiers (visual separator)
tier_gap = 1.5;

// Barrel vault roof
roof_r = top_d / 2;          // Half-cylinder spans full top tier depth
roof_len = top_w + 4;        // Slightly longer than top tier

// Flaps (front/back)
flap_ext = 10;
flap_t = 2.5;

// Window decoration — rectangular panel frames inset into each face
win_w_ratio = 0.55;          // Window width as fraction of column width
win_h_ratio = 0.55;          // Window height as fraction of tier height
win_inset = 0.8;             // How deep the decorative frame is inset

// Snap-fit
snap_lip = 1.5;              // Height of snap rim/groove
snap_clearance = 0.25;       // Tolerance for a firm click fit

// Base plate thickness
base_t = 2.0;

// ─── Single tier: solid shell with window cutouts ───

module tier_shell(w, d, h, cols_front, cols_side) {
    difference() {
        // Outer solid block
        cube([w, d, h], center=true);

        // Hollow interior (keep wall thickness on all sides)
        cube([w - wall*2, d - wall*2, h + 2], center=true);

        // Window cutouts — front and back faces (Y sides)
        col_w = (w - wall*2) / cols_front;
        for (c = [0:cols_front-1]) {
            cx = -(w - wall*2)/2 + col_w * (c + 0.5);
            ww = col_w * win_w_ratio;
            wh = h * win_h_ratio;
            translate([cx, 0, 0])
                cube([ww, d + 2, wh], center=true);
        }

        // Window cutouts — left and right faces (X sides)
        col_d = (d - wall*2) / cols_side;
        for (c = [0:cols_side-1]) {
            cy = -(d - wall*2)/2 + col_d * (c + 0.5);
            ww = col_d * win_w_ratio;
            wh = h * win_h_ratio;
            translate([0, cy, 0])
                cube([w + 2, ww, wh], center=true);
        }
    }

    // Decorative inset frame around each window (raised outline on outer face)
    // Front/back
    for (sy = [-1, 1])
    for (c = [0:cols_front-1]) {
        col_w = (w - wall*2) / cols_front;
        cx = -(w - wall*2)/2 + col_w * (c + 0.5);
        ww = col_w * win_w_ratio;
        wh = h * win_h_ratio;
        translate([cx, sy * (d/2 - win_inset/2), 0])
            difference() {
                cube([ww + 3, win_inset, wh + 3], center=true);
                cube([ww, win_inset + 1, wh], center=true);
            }
    }
    // Left/right
    for (sx = [-1, 1])
    for (c = [0:cols_side-1]) {
        col_d = (d - wall*2) / cols_side;
        cy = -(d - wall*2)/2 + col_d * (c + 0.5);
        ww = col_d * win_w_ratio;
        wh = h * win_h_ratio;
        translate([sx * (w/2 - win_inset/2), cy, 0])
            difference() {
                cube([win_inset, ww + 3, wh + 3], center=true);
                cube([win_inset + 1, ww, wh], center=true);
            }
    }
}

// ─── PART 1: Body — 3 stacked tiers + snap rim ───

module body() {
    // Bottom tier — solid bottom (closes the lantern)
    z_bot = 0;
    difference() {
        translate([0, 0, z_bot])
            tier_shell(bot_w, bot_d, bot_h, 1, 1);
        // Re-close the bottom floor (fill in the hollow cut at the bottom)
    }
    // Solid floor plate at the bottom
    translate([0, 0, z_bot - bot_h/2 + cap_t/2])
        cube([bot_w - wall*2 + 0.1, bot_d - wall*2 + 0.1, cap_t], center=true);

    // Middle tier
    z_mid = z_bot + bot_h/2 + tier_gap + mid_h/2;
    translate([0, 0, z_mid])
        tier_shell(mid_w, mid_d, mid_h, 2, 1);

    // Top tier
    z_top = z_mid + mid_h/2 + tier_gap + top_h/2;
    translate([0, 0, z_top])
        tier_shell(top_w, top_d, top_h, 3, 2);

    // Tier connectors (thin posts between tiers so they're one piece)
    // Bottom → middle (4 corner posts)
    z_connect1 = z_bot + bot_h/2 + tier_gap/2;
    for (sx = [-1, 1]) for (sy = [-1, 1])
        translate([sx * bot_w/2 * 0.7, sy * bot_d/2 * 0.7, z_connect1])
            cube([3, 3, tier_gap + 0.2], center=true);

    // Middle → top
    z_connect2 = z_mid + mid_h/2 + tier_gap/2;
    for (sx = [-1, 1]) for (sy = [-1, 1])
        translate([sx * mid_w/2 * 0.7, sy * mid_d/2 * 0.7, z_connect2])
            cube([3, 3, tier_gap + 0.2], center=true);

    // Snap-fit rim on top of top tier — male ridge
    z_rim = z_top + top_h/2 + snap_lip/2;
    translate([0, 0, z_rim])
        difference() {
            cube([top_w - 1.5, top_d - 1.5, snap_lip], center=true);
            cube([top_w - 1.5 - wall*2, top_d - 1.5 - wall*2, snap_lip + 1],
                 center=true);
        }
}

// ─── PART 2: Roof — barrel vault with closed ends ───

module roof() {
    // Base plate
    translate([0, 0, base_t/2])
        cube([top_w + 4, top_d + 4, base_t], center=true);

    // Snap-fit groove cut into underside of base plate (female)
    // (Implemented as a separate difference at assembly level below)

    // Barrel vault — hollow half-cylinder
    translate([0, 0, base_t])
        difference() {
            // Outer shell
            union() {
                // Cylinder body
                rotate([0, 90, 0])
                    cylinder(r=roof_r, h=roof_len, center=true);
                // Closed end caps — solid disks at each end, flush with the outer cylinder
                for (sx = [-1, 1])
                    translate([sx * roof_len/2, 0, 0])
                        rotate([0, 90, 0])
                            cylinder(r=roof_r, h=wall, center=true);
            }
            // Inner hollow — ONLY hollow the body, not the end caps (stops slightly short of the ends)
            rotate([0, 90, 0])
                cylinder(r=roof_r - wall, h=roof_len - wall*2 - 1, center=true);
            // Cut the bottom half in world Z
            translate([0, 0, -roof_r])
                cube([roof_len + 20, roof_r * 3, roof_r * 2], center=true);
        }

    // Flaps — front/back, attached to base plate
    for (sy = [-1, 1])
        translate([0, sy * (top_d/2 + 2 + flap_ext/2), flap_t/2])
            cube([top_w + 4, flap_ext, flap_t], center=true);

    // Hanging loop on top of ridge
    translate([0, 0, base_t + roof_r + 3])
        rotate([90, 0, 0])
            difference() {
                cylinder(r=5, h=4, center=true);
                cylinder(r=3, h=5, center=true);
            }
}

module roof_with_groove() {
    difference() {
        roof();
        // Snap-fit groove — inverse of the body's rim, with clearance
        translate([0, 0, -snap_lip/2])
            difference() {
                cube([top_w - 1.5 + snap_clearance*2,
                      top_d - 1.5 + snap_clearance*2,
                      snap_lip + 0.2],
                     center=true);
                cube([top_w - 1.5 - wall*2 - snap_clearance*2,
                      top_d - 1.5 - wall*2 - snap_clearance*2,
                      snap_lip + 2],
                     center=true);
            }
    }
}

// ─── Assembly preview ───

module assembled() {
    body();
    z_body_top = bot_h/2 + tier_gap + mid_h + tier_gap + top_h + snap_lip;
    translate([0, 0, z_body_top])
        roof_with_groove();
}

// ─── Export selector ───

if (part == "both") {
    assembled();
} else if (part == "body") {
    body();
} else if (part == "roof") {
    roof_with_groove();
}
