// Ottoman Hanging Lantern — 2-part snap-fit design, 3D printable
//
// PART 1 (body): 3 stepped tiers with solid tapered transitions between them
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
tier_gap = 1.5;

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

// Total body height (for print flip calculation)
body_total_h = bot_h + tier_gap + mid_h + tier_gap + top_h + snap_lip;

// ─── Single tier: solid shell with window cutouts ───

module tier_shell(w, d, h, cols_front, cols_side) {
    difference() {
        cube([w, d, h], center=true);
        cube([w - wall*2, d - wall*2, h + 2], center=true);

        col_w = (w - wall*2) / cols_front;
        for (c = [0:cols_front-1]) {
            cx = -(w - wall*2)/2 + col_w * (c + 0.5);
            ww = col_w * win_w_ratio;
            wh = h * win_h_ratio;
            translate([cx, 0, 0])
                cube([ww, d + 2, wh], center=true);
        }

        col_d = (d - wall*2) / cols_side;
        for (c = [0:cols_side-1]) {
            cy = -(d - wall*2)/2 + col_d * (c + 0.5);
            ww = col_d * win_w_ratio;
            wh = h * win_h_ratio;
            translate([0, cy, 0])
                cube([w + 2, ww, wh], center=true);
        }
    }

    // Decorative inset frames — front/back
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
    // Decorative inset frames — left/right
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

// ─── PART 1: Body — 3 stacked tiers with solid tapered transitions ───

module body() {
    // Bottom tier — bottom face at z=0
    translate([0, 0, bot_h/2])
        tier_shell(bot_w, bot_d, bot_h, 1, 1);

    // Solid floor plate at the very bottom
    translate([0, 0, cap_t/2])
        cube([bot_w - wall*2 + 0.1, bot_d - wall*2 + 0.1, cap_t], center=true);

    // Solid tapered transition: bottom → middle (hull = no air gap)
    hull() {
        translate([0, 0, bot_h])
            cube([bot_w, bot_d, 0.01], center=true);
        translate([0, 0, bot_h + tier_gap])
            cube([mid_w, mid_d, 0.01], center=true);
    }

    // Middle tier
    translate([0, 0, bot_h + tier_gap + mid_h/2])
        tier_shell(mid_w, mid_d, mid_h, 2, 1);

    // Solid tapered transition: middle → top
    z_mid_top = bot_h + tier_gap + mid_h;
    hull() {
        translate([0, 0, z_mid_top])
            cube([mid_w, mid_d, 0.01], center=true);
        translate([0, 0, z_mid_top + tier_gap])
            cube([top_w, top_d, 0.01], center=true);
    }

    // Top tier
    z_top_base = bot_h + tier_gap + mid_h + tier_gap;
    translate([0, 0, z_top_base + top_h/2])
        tier_shell(top_w, top_d, top_h, 3, 2);

    // Snap-fit rim on top of top tier — male ridge
    z_rim = z_top_base + top_h + snap_lip/2;
    translate([0, 0, z_rim])
        difference() {
            cube([top_w - 1.5, top_d - 1.5, snap_lip], center=true);
            cube([top_w - 1.5 - wall*2, top_d - 1.5 - wall*2, snap_lip + 1],
                 center=true);
        }
}

// Body flipped for printing — largest tier flat on build plate, no overhangs
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
    body_print();    // Flipped for printing
} else if (part == "roof") {
    roof_with_groove();
}
