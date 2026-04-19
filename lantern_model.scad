// Ottoman Hanging Lantern — 2-part snap-fit design for 3D printing
// Part 1 (body): 3 tiers — print UPSIDE DOWN (top tier on bed, no overhangs)
// Part 2 (roof): Barrel vault + flaps — print RIGHT-SIDE UP (minimal bridging)
//
// Set `part` below to export each piece for slicing:
//   "both"  = preview assembled (default)
//   "body"  = just the body (mirror it in slicer to flip upside down)
//   "roof"  = just the roof
//
// Snap-fit: body has a rim around the top edge; roof has a matching groove.

$fn = 48;

part = "both";   // "both", "body", or "roof"

// ─── Parameters ───

frame = 2.5;
wall = 2.4;

// Tier outer dimensions
top_w = 72;  top_d = 52;  top_h = 32;
mid_w = 50;  mid_d = 40;  mid_h = 24;
bot_w = 32;  bot_d = 28;  bot_h = 18;

// Barrel vault roof — half-cylinder spanning full top tier depth
roof_r = top_d / 2;   // = 26, so vault diameter equals top tier depth

// Front/back flaps on top tier
flap_ext = 10;
flap_t = 3;

// Snap-fit dimensions
snap_lip = 1.2;      // Height of snap lip
snap_clearance = 0.3; // Tolerance for fit

chain_h = 20;
bracket_reach = 80;

// ─── Glass face with rectangular window panels ───

module glass_face(w, h, cols) {
    color([0.92, 0.9, 0.85])
        cube([w, 1.2, h], center=true);

    color([0.12, 0.1, 0.08]) {
        translate([0, 0, h/2])  cube([w + 1, frame, frame], center=true);
        translate([0, 0, -h/2]) cube([w + 1, frame, frame], center=true);
        translate([w/2, 0, 0])  cube([frame, frame, h + 1], center=true);
        translate([-w/2, 0, 0]) cube([frame, frame, h + 1], center=true);

        for (c = [1:cols-1]) {
            xpos = -w/2 + c * (w / cols);
            translate([xpos, 0, 0])
                cube([frame * 0.7, frame, h], center=true);
        }

        col_w = w / cols;
        for (c = [0:cols-1]) {
            cx = -w/2 + col_w * (c + 0.5);
            win_w = col_w * 0.55;
            win_h = h * 0.55;
            translate([cx, 0, win_h/2])
                cube([win_w, frame * 0.6, frame * 0.6], center=true);
            translate([cx, 0, -win_h/2])
                cube([win_w, frame * 0.6, frame * 0.6], center=true);
            translate([cx - win_w/2, 0, 0])
                cube([frame * 0.6, frame * 0.6, win_h], center=true);
            translate([cx + win_w/2, 0, 0])
                cube([frame * 0.6, frame * 0.6, win_h], center=true);
        }
    }
}

// ─── Tier ───

module tier(w, d, h, cols_front, cols_side) {
    gw = w - frame * 2;
    gd = d - frame * 2;
    gh = h - frame * 2;

    translate([0, d/2, 0])  glass_face(gw, gh, cols_front);
    translate([0, -d/2, 0]) glass_face(gw, gh, cols_front);
    translate([w/2, 0, 0])  rotate([0, 0, 90]) glass_face(gd, gh, cols_side);
    translate([-w/2, 0, 0]) rotate([0, 0, 90]) glass_face(gd, gh, cols_side);

    color([0.12, 0.1, 0.08])
    for (sx = [-1, 1])
        for (sy = [-1, 1])
            translate([sx * w/2, sy * d/2, 0])
                cube([frame, frame, h], center=true);

    color([0.15, 0.12, 0.1]) {
        translate([0, 0, h/2])
            cube([w + 2, d + 2, frame], center=true);
        translate([0, 0, -h/2])
            cube([w + 2, d + 2, frame], center=true);
    }
}

// ─── PART 1: Body (3 tiers + snap rim on top) ───
// Print upside down: top tier flat on bed, tiers get narrower going up = no overhangs

module body() {
    z_bot = 0;
    z_mid = z_bot + bot_h/2 + mid_h/2 + 1;
    z_top = z_mid + mid_h/2 + top_h/2 + 1;

    translate([0, 0, z_bot])
        tier(bot_w, bot_d, bot_h, 1, 1);
    translate([0, 0, z_mid])
        tier(mid_w, mid_d, mid_h, 2, 2);
    translate([0, 0, z_top])
        tier(top_w, top_d, top_h, 3, 2);

    // Snap-fit rim on top of top tier (male part — ridge sticks up)
    color([0.15, 0.12, 0.1])
    translate([0, 0, z_top + top_h/2 + snap_lip/2])
        difference() {
            cube([top_w - 2, top_d - 2, snap_lip], center=true);
            cube([top_w - 2 - wall*2, top_d - 2 - wall*2, snap_lip + 1], center=true);
        }
}

// ─── PART 2: Roof (barrel vault + flaps + snap groove + hanging loop) ───
// Print right-side up: base plate on bed, vault bridges across the top

module roof() {
    color([0.15, 0.12, 0.1]) {
        // Base plate — sits flat ABOVE the top tier, not overlapping it
        translate([0, 0, 0])
            cube([top_w + 4, top_d + 4, frame], center=true);

        // Half-cylinder vault — sits ON TOP of base plate, centered at top of base plate
        // Top half only (top_r radius, starts at z = frame/2 upward)
        translate([0, 0, frame/2])
            difference() {
                rotate([0, 90, 0])
                    cylinder(r=roof_r, h=top_w + 4, center=true);
                rotate([0, 90, 0])
                    cylinder(r=roof_r - wall, h=top_w + 6, center=true);
                // Cut everything below z=0 local (below base plate top)
                translate([0, 0, -roof_r])
                    cube([top_w + 10, roof_r * 3, roof_r * 2], center=true);
            }

        // End caps (solid semicircles — cut bottom half in world Z)
        for (sx = [-1, 1])
            translate([sx * (top_w/2 + 1.5), 0, frame/2])
                difference() {
                    rotate([0, 90, 0])
                        cylinder(r=roof_r, h=frame, center=true);
                    translate([0, 0, -roof_r])
                        cube([frame + 2, roof_r * 3, roof_r * 2], center=true);
                }

        // Front/back flaps — on the base plate, outside the top tier footprint
        for (sy = [-1, 1])
            translate([0, sy * (top_d/2 + 2 + flap_ext/2), 0])
                cube([top_w + 4, flap_ext, flap_t], center=true);

        // Hanging loop on top
        translate([0, 0, frame/2 + roof_r + 2])
            rotate([90, 0, 0])
                difference() {
                    cylinder(r=5, h=4, center=true);
                    cylinder(r=3, h=5, center=true);
                }
    }

    // Snap-fit groove cut into underside of base plate
    color([0.15, 0.12, 0.1])
    translate([0, 0, -frame/2 - snap_lip/2])
        difference() {
            cube([top_w, top_d, snap_lip], center=true);
            cube([top_w - wall*2 - snap_clearance*2, top_d - wall*2 - snap_clearance*2, snap_lip + 1], center=true);
        }
}

// ─── Assembly ───

module assembled() {
    z_top_surface = bot_h/2 + mid_h/2 + 1 + mid_h/2 + top_h/2 + 1 + top_h/2;

    body();
    translate([0, 0, z_top_surface + frame/2 + snap_lip])
        roof();
}

// ─── Export selector ───

if (part == "both") {
    assembled();
} else if (part == "body") {
    // For printing: flip upside down in your slicer
    body();
} else if (part == "roof") {
    roof();
}
