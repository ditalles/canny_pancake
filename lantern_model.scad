// Ottoman/Turkish Hanging Lantern — 3-tier, barrel vault top with side flaps
// White glass faces with dark metal frames and rectangular window panels

$fn = 50;

// ─── Parameters ───

frame = 2.5;

// Three tiers — decreasing in size downward
top_w = 72;  top_d = 52;  top_h = 32;    // widest, tallest
mid_w = 50;  mid_d = 40;  mid_h = 24;    // middle
bot_w = 32;  bot_d = 28;  bot_h = 18;    // smallest

// Barrel vault roof
roof_r = 28;        // Covers full depth of top tier (top_d/2 = 26)

// Side flap/wing on top tier
flap_w = 14;        // How far flap extends beyond top tier sides
flap_d = 38;        // Depth of flap
flap_t = 2.5;       // Thickness

chain_h = 25;
bracket_reach = 80;

// ─── Glass face with rectangular window panels ───
// Each panel has a rectangular frame inside it (decorative window motif)

module glass_face(w, h, cols) {
    // Glass pane (single background)
    color([0.92, 0.9, 0.85])
        cube([w, 1.2, h], center=true);

    color([0.12, 0.1, 0.08]) {
        // Outer frame
        translate([0, 0, h/2])  cube([w + 1, frame, frame], center=true);
        translate([0, 0, -h/2]) cube([w + 1, frame, frame], center=true);
        translate([w/2, 0, 0])  cube([frame, frame, h + 1], center=true);
        translate([-w/2, 0, 0]) cube([frame, frame, h + 1], center=true);

        // Vertical dividers between columns
        for (c = [1:cols-1]) {
            xpos = -w/2 + c * (w / cols);
            translate([xpos, 0, 0])
                cube([frame * 0.7, frame, h], center=true);
        }

        // Rectangular window frame inside each column
        col_w = w / cols;
        for (c = [0:cols-1]) {
            cx = -w/2 + col_w * (c + 0.5);
            // Inner rectangle frame
            win_w = col_w * 0.55;
            win_h = h * 0.55;
            // Top bar
            translate([cx, 0, win_h/2])
                cube([win_w, frame * 0.6, frame * 0.6], center=true);
            // Bottom bar
            translate([cx, 0, -win_h/2])
                cube([win_w, frame * 0.6, frame * 0.6], center=true);
            // Left side
            translate([cx - win_w/2, 0, 0])
                cube([frame * 0.6, frame * 0.6, win_h], center=true);
            // Right side
            translate([cx + win_w/2, 0, 0])
                cube([frame * 0.6, frame * 0.6, win_h], center=true);
        }
    }
}

// ─── Complete tier ───

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

// ─── Barrel vault roof ───

module barrel_roof(w, d) {
    color([0.15, 0.12, 0.1]) {
        // Base plate
        translate([0, 0, -1])
            cube([w + 4, d + 4, frame], center=true);

        // Half-cylinder vault
        difference() {
            rotate([0, 90, 0])
                cylinder(r=roof_r, h=w + 4, center=true);
            rotate([0, 90, 0])
                cylinder(r=roof_r - frame, h=w + 6, center=true);
            translate([0, 0, -roof_r])
                cube([w + 10, roof_r * 3, roof_r * 2], center=true);
        }

        // End caps (front/back)
        for (sx = [-1, 1])
            translate([sx * (w/2 + 1.5), 0, 0])
            rotate([0, 90, 0])
                difference() {
                    cylinder(r=roof_r, h=frame, center=true);
                    translate([0, 0, -roof_r])
                        cube([roof_r * 3, roof_r * 3, roof_r * 2], center=true);
                }
    }
}

// ─── Side flap/wing that sticks out from sides of top tier ───

module side_flaps(w, d) {
    color([0.15, 0.12, 0.1])
    for (sx = [-1, 1])
        translate([sx * (w/2 + flap_w/2), 0, 0])
            cube([flap_w, flap_d, flap_t], center=true);
}

// ─── Chain ───

module chain() {
    color([0.18, 0.15, 0.12])
    for (i = [0:floor(chain_h/6)-1]) {
        translate([0, 0, i * 6]) {
            if (i % 2 == 0)
                difference() {
                    cube([2, 4, 6], center=true);
                    cube([0.8, 2.5, 4.5], center=true);
                }
            else
                difference() {
                    cube([4, 2, 6], center=true);
                    cube([2.5, 0.8, 4.5], center=true);
                }
        }
    }
}

// ─── Curved bracket arm ───

module bracket() {
    color([0.12, 0.1, 0.08]) {
        translate([-bracket_reach, 0, 10])
            cube([6, 30, 45], center=true);

        for (t = [0:3:90]) {
            hull() {
                translate([-bracket_reach + 6 + bracket_reach * sin(t), 0,
                           25 - bracket_reach * 0.3 * (1 - cos(t))])
                    sphere(r=2.5, $fn=10);
                translate([-bracket_reach + 6 + bracket_reach * sin(t+3), 0,
                           25 - bracket_reach * 0.3 * (1 - cos(t+3))])
                    sphere(r=2.5, $fn=10);
            }
        }

        translate([-bracket_reach + 25, 0, 15])
        rotate([90, 0, 0])
            difference() {
                cylinder(r=10, h=3, center=true, $fn=30);
                cylinder(r=7, h=4, center=true, $fn=30);
            }

        translate([5, 0, -2])
        rotate([90, 0, 0])
            difference() {
                cylinder(r=5, h=3, center=true, $fn=20);
                cylinder(r=3.5, h=4, center=true, $fn=20);
                translate([0, -5, 0])
                    cube([12, 6, 5], center=true);
            }
    }
}

// ─── Full assembly ───

module lantern() {
    z_bot = 0;
    z_mid = z_bot + bot_h/2 + mid_h/2 + 1;
    z_top = z_mid + mid_h/2 + top_h/2 + 1;
    z_flaps = z_top + top_h/2 - 1;
    z_roof = z_top + top_h/2;
    z_chain = z_roof + roof_r + 2;
    z_bracket = z_chain + chain_h;

    // Bottom tier (1 col, no inner window)
    translate([0, 0, z_bot])
        tier(bot_w, bot_d, bot_h, 1, 1);

    // Middle tier (2 cols front, 2 cols side)
    translate([0, 0, z_mid])
        tier(mid_w, mid_d, mid_h, 2, 2);

    // Top tier (3 cols front, 2 cols side)
    translate([0, 0, z_top])
        tier(top_w, top_d, top_h, 3, 2);

    // Side flaps/wings on top tier
    translate([0, 0, z_flaps])
        side_flaps(top_w, top_d);

    // Barrel vault roof
    translate([0, 0, z_roof])
        barrel_roof(top_w, top_d);

    // Chain
    translate([0, 0, z_chain])
        chain();

    // Bracket
    translate([0, 0, z_bracket])
        bracket();
}

lantern();
