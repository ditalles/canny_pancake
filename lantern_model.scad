// Ottoman Hanging Lantern — 2-part snap-fit design, 3D printable
//
// PART 1 (body): Solid stepped pyramid with 3 tiers, fully closed chambers
//                Exported UPSIDE DOWN — largest tier on build plate, no overhangs
// PART 2 (roof): Barrel vault with closed ends, flaps, and hanging loop
//                Exported base-plate-down
//
// HOW TO EXPORT STLs:
//   1. Set `part = "body"` → Render (F6) → Export STL (pre-oriented for printing)
//   2. Set `part = "roof"` → Render (F6) → Export STL (pre-oriented for printing)
//   3. Slice with 0.2mm layer, 2-3 perimeters, 15-20% infill, no supports

$fn = 60;

part = "both";   // "both", "body", or "roof"

// ─── Parameters ───

wall = 2.0;

// Tier outer dimensions (width × depth × height)
top_w = 72;  top_d = 52;  top_h = 30;
mid_w = 50;  mid_d = 40;  mid_h = 22;
bot_w = 32;  bot_d = 26;  bot_h = 16;

cap_t = 2.0;
sep_t = 2.5;

// Barrel vault roof
roof_r = top_d / 2;
roof_len = top_w + 4;

// Flaps (front/back)
flap_ext = 10;
flap_t = 2.5;

// Window decoration
win_w_ratio = 0.55;
win_h_ratio = 0.55;
win_inset = 0.8;

// Snap-fit
snap_lip = 1.5;
snap_clearance = 0.25;

// Base plate thickness
base_t = 2.0;

// Z coordinates for body tiers
z1 = cap_t;                        // floor top / bot tier start
z2 = z1 + bot_h;                   // bot tier top / sep1 start
z3 = z2 + sep_t;                   // sep1 top / mid tier start
z4 = z3 + mid_h;                   // mid tier top / sep2 start
z5 = z4 + sep_t;                   // sep2 top / top tier start
z6 = z5 + top_h;                   // top tier top / ceiling start
z7 = z6 + cap_t;                   // ceiling top / snap rim start
z8 = z7 + snap_lip;                // snap rim top

body_total_h = z8;

// ─── Window cutout helper (used inside difference) ───

module win_cuts_front_back(w, d, h, cols, z_center) {
    col_w = (w - wall*2) / cols;
    for (c = [0:cols-1]) {
        cx = -(w - wall*2)/2 + col_w * (c + 0.5);
        ww = col_w * win_w_ratio;
        wh = h * win_h_ratio;
        translate([cx, 0, z_center])
            cube([ww, d + 2, wh], center=true);
    }
}

module win_cuts_left_right(w, d, h, cols, z_center) {
    col_d = (d - wall*2) / cols;
    for (c = [0:cols-1]) {
        cy = -(d - wall*2)/2 + col_d * (c + 0.5);
        ww = col_d * win_w_ratio;
        wh = h * win_h_ratio;
        translate([0, cy, z_center])
            cube([w + 2, ww, wh], center=true);
    }
}

// ─── Window frame helper (added as positive geometry) ───

module win_frames_front_back(w, d, h, cols, z_center) {
    for (sy = [-1, 1])
    for (c = [0:cols-1]) {
        col_w = (w - wall*2) / cols;
        cx = -(w - wall*2)/2 + col_w * (c + 0.5);
        ww = col_w * win_w_ratio;
        wh = h * win_h_ratio;
        translate([cx, sy * (d/2 - win_inset/2), z_center])
            difference() {
                cube([ww + 3, win_inset, wh + 3], center=true);
                cube([ww, win_inset + 1, wh], center=true);
            }
    }
}

module win_frames_left_right(w, d, h, cols, z_center) {
    for (sx = [-1, 1])
    for (c = [0:cols-1]) {
        col_d = (d - wall*2) / cols;
        cy = -(d - wall*2)/2 + col_d * (c + 0.5);
        ww = col_d * win_w_ratio;
        wh = h * win_h_ratio;
        translate([sx * (w/2 - win_inset/2), cy, z_center])
            difference() {
                cube([win_inset, ww + 3, wh + 3], center=true);
                cube([win_inset + 1, ww, wh], center=true);
            }
    }
}

// ─── PART 1: Body — single solid with precisely cut chambers ───

module body() {
    difference() {
        // === Outer solid step-pyramid ===
        union() {
            // Floor plate (bot dimensions)
            translate([0, 0, cap_t/2])
                cube([bot_w, bot_d, cap_t], center=true);
            // Bot tier solid block
            translate([0, 0, z1 + bot_h/2])
                cube([bot_w, bot_d, bot_h], center=true);
            // Separator plate 1 (mid dimensions)
            translate([0, 0, z2 + sep_t/2])
                cube([mid_w, mid_d, sep_t], center=true);
            // Mid tier solid block
            translate([0, 0, z3 + mid_h/2])
                cube([mid_w, mid_d, mid_h], center=true);
            // Separator plate 2 (top dimensions)
            translate([0, 0, z4 + sep_t/2])
                cube([top_w, top_d, sep_t], center=true);
            // Top tier solid block
            translate([0, 0, z5 + top_h/2])
                cube([top_w, top_d, top_h], center=true);
            // Ceiling plate (closes top tier)
            translate([0, 0, z6 + cap_t/2])
                cube([top_w, top_d, cap_t], center=true);
            // Snap-fit rim
            translate([0, 0, z7 + snap_lip/2])
                cube([top_w - 1.5, top_d - 1.5, snap_lip], center=true);
        }

        // === Hollow out tier interiors (precisely bounded) ===
        // Bot tier cavity — between floor and sep1 only
        translate([0, 0, z1 + bot_h/2])
            cube([bot_w - wall*2, bot_d - wall*2, bot_h], center=true);
        // Mid tier cavity — between sep1 and sep2 only
        translate([0, 0, z3 + mid_h/2])
            cube([mid_w - wall*2, mid_d - wall*2, mid_h], center=true);
        // Top tier cavity — between sep2 and ceiling only
        translate([0, 0, z5 + top_h/2])
            cube([top_w - wall*2, top_d - wall*2, top_h], center=true);
        // Snap rim inner hollow
        translate([0, 0, z7 + snap_lip/2])
            cube([top_w - 1.5 - wall*2, top_d - 1.5 - wall*2, snap_lip + 1],
                 center=true);

        // === Window cutouts (through walls only, not plates) ===
        // Bot tier windows
        win_cuts_front_back(bot_w, bot_d, bot_h, 1, z1 + bot_h/2);
        win_cuts_left_right(bot_w, bot_d, bot_h, 1, z1 + bot_h/2);
        // Mid tier windows
        win_cuts_front_back(mid_w, mid_d, mid_h, 2, z3 + mid_h/2);
        win_cuts_left_right(mid_w, mid_d, mid_h, 1, z3 + mid_h/2);
        // Top tier windows
        win_cuts_front_back(top_w, top_d, top_h, 3, z5 + top_h/2);
        win_cuts_left_right(top_w, top_d, top_h, 2, z5 + top_h/2);
    }

    // === Decorative window frames (positive geometry) ===
    win_frames_front_back(bot_w, bot_d, bot_h, 1, z1 + bot_h/2);
    win_frames_left_right(bot_w, bot_d, bot_h, 1, z1 + bot_h/2);
    win_frames_front_back(mid_w, mid_d, mid_h, 2, z3 + mid_h/2);
    win_frames_left_right(mid_w, mid_d, mid_h, 1, z3 + mid_h/2);
    win_frames_front_back(top_w, top_d, top_h, 3, z5 + top_h/2);
    win_frames_left_right(top_w, top_d, top_h, 2, z5 + top_h/2);
}

// Body flipped for printing — largest tier flat on build plate
module body_print() {
    translate([0, 0, body_total_h])
        mirror([0, 0, 1])
            body();
}

// ─── PART 2: Roof — barrel vault with closed ends ───

module roof() {
    // Base plate
    translate([0, 0, base_t/2])
        cube([top_w + 4, top_d + 4, base_t], center=true);

    // Barrel vault — hollow half-cylinder with closed ends
    translate([0, 0, base_t])
        difference() {
            union() {
                rotate([0, 90, 0])
                    cylinder(r=roof_r, h=roof_len, center=true);
                for (sx = [-1, 1])
                    translate([sx * roof_len/2, 0, 0])
                        rotate([0, 90, 0])
                            cylinder(r=roof_r, h=wall, center=true);
            }
            rotate([0, 90, 0])
                cylinder(r=roof_r - wall, h=roof_len - wall*2 - 1, center=true);
            // Cut bottom half
            translate([0, 0, -roof_r])
                cube([roof_len + 20, roof_r * 3, roof_r * 2], center=true);
        }

    // Flaps — front/back
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
        // Snap-fit groove — female, mates with body's rim
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
    translate([0, 0, body_total_h])
        roof_with_groove();
}

// ─── Export selector ───

if (part == "both") {
    assembled();
} else if (part == "body") {
    body_print();
} else if (part == "roof") {
    roof_with_groove();
}
