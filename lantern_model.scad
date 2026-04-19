// Ottoman/Turkish Hanging Lantern — 4-sided, 3-tier, barrel-vault top
// White glass faces with dark metal mullion grids
// Render: F5 preview, F6 full render, export STL for 3D printing

$fn = 50;

// ─── Parameters ───

frame = 2.5;

// Tier sizes — top is widest/squattest, bottom is smallest
top_w = 78;   top_d = 55;   top_h = 38;   // wide & short
mid_w = 52;   mid_d = 40;   mid_h = 38;
bot_w = 32;   bot_d = 26;   bot_h = 16;   // small

band = 4;
roof_r = 29;        // Covers full depth of top tier (top_d/2 ≈ 27.5)

chain_h = 12;       // Short chain
bracket_reach = 80;

// ─── Glass face with mullion grid ───

module glass_face(w, h, cols, rows) {
    color([0.92, 0.9, 0.85])
        cube([w, 1.2, h], center=true);

    color([0.12, 0.1, 0.08]) {
        // Border
        translate([0, 0, h/2])  cube([w + 1, frame, frame], center=true);
        translate([0, 0, -h/2]) cube([w + 1, frame, frame], center=true);
        translate([w/2, 0, 0])  cube([frame, frame, h + 1], center=true);
        translate([-w/2, 0, 0]) cube([frame, frame, h + 1], center=true);

        // Vertical mullions
        for (c = [1:cols-1]) {
            xpos = -w/2 + c * (w / cols);
            translate([xpos, 0, 0])
                cube([frame * 0.7, frame, h], center=true);
        }
        // Horizontal mullions
        if (rows > 1)
        for (r = [1:rows-1]) {
            zpos = -h/2 + r * (h / rows);
            translate([0, 0, zpos])
                cube([w, frame, frame * 0.7], center=true);
        }
    }
}

// ─── Complete tier ───

module tier(w, d, h, cols, rows) {
    gw = w - frame * 2;
    gd = d - frame * 2;
    gh = h - frame * 2;

    // 4 glass faces
    translate([0, d/2, 0])
        glass_face(gw, gh, cols, rows);
    translate([0, -d/2, 0])
        glass_face(gw, gh, cols, rows);
    translate([w/2, 0, 0])
        rotate([0, 0, 90])
            glass_face(gd, gh, cols, rows);
    translate([-w/2, 0, 0])
        rotate([0, 0, 90])
            glass_face(gd, gh, cols, rows);

    // Corner posts
    color([0.12, 0.1, 0.08])
    for (sx = [-1, 1])
        for (sy = [-1, 1])
            translate([sx * w/2, sy * d/2, 0])
                cube([frame, frame, h], center=true);

    // Top/bottom cap plates
    color([0.15, 0.12, 0.1]) {
        translate([0, 0, h/2])
            cube([w + 2, d + 2, frame], center=true);
        translate([0, 0, -h/2])
            cube([w + 2, d + 2, frame], center=true);
    }
}

// ─── Spacer band ───

module spacer(w_top, d_top, w_bot, d_bot) {
    color([0.15, 0.12, 0.1])
        hull() {
            translate([0, 0, band/2])
                cube([w_top + 2, d_top + 2, 0.1], center=true);
            translate([0, 0, -band/2])
                cube([w_bot + 2, d_bot + 2, 0.1], center=true);
        }
}

// ─── Barrel vault roof ───

module barrel_roof(w, d) {
    color([0.15, 0.12, 0.1]) {
        // Base plate
        translate([0, 0, -1])
            cube([w + 4, d + 4, frame], center=true);

        // Half-cylinder vault — sized to match the depth
        difference() {
            rotate([0, 90, 0])
                cylinder(r=roof_r, h=w + 4, center=true);
            rotate([0, 90, 0])
                cylinder(r=roof_r - frame, h=w + 6, center=true);
            translate([0, 0, -roof_r])
                cube([w + 10, roof_r * 3, roof_r * 2], center=true);
        }

        // Ridge bar
        translate([0, 0, roof_r - 1])
            cube([w + 6, frame, frame], center=true);

        // End caps
        for (sx = [-1, 1])
            translate([sx * (w/2 + 1.5), 0, 0])
            rotate([0, 90, 0])
                difference() {
                    cylinder(r=roof_r, h=frame, center=true);
                    cylinder(r=roof_r - frame * 2, h=frame + 1, center=true);
                    translate([0, 0, -roof_r])
                        cube([roof_r * 3, roof_r * 3, roof_r * 2], center=true);
                }

        // Glass in end caps
        for (sx = [-1, 1])
            color([0.92, 0.9, 0.85])
            translate([sx * (w/2 + 1.5), 0, 0])
            rotate([0, 90, 0])
                difference() {
                    cylinder(r=roof_r - frame * 2 - 0.5, h=0.8, center=true);
                    translate([0, 0, -roof_r])
                        cube([roof_r * 3, roof_r * 3, roof_r * 2], center=true);
                }
    }

}

// ─── Short chain ───

module chain() {
    color([0.18, 0.15, 0.12])
    for (i = [0:floor(chain_h/7)-1]) {
        translate([0, 0, i * 7]) {
            if (i % 2 == 0)
                difference() {
                    cube([2, 4, 7], center=true);
                    cube([0.8, 2.5, 5], center=true);
                }
            else
                difference() {
                    cube([4, 2, 7], center=true);
                    cube([2.5, 0.8, 5], center=true);
                }
        }
    }
}

// ─── Curved bracket arm — curves down from wall ───

module bracket() {
    color([0.12, 0.1, 0.08]) {
        // Wall plate
        translate([-bracket_reach, 0, 10])
            cube([6, 30, 45], center=true);

        // Curved arm — arcs down from wall to lantern
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

        // Scroll decoration near wall
        translate([-bracket_reach + 25, 0, 15])
        rotate([90, 0, 0])
            difference() {
                cylinder(r=10, h=3, center=true, $fn=30);
                cylinder(r=7, h=4, center=true, $fn=30);
            }

        // Hook at end
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
    z3 = 0;
    z_sp2 = z3 + bot_h/2 + band/2;
    z2 = z_sp2 + band/2 + mid_h/2;
    z_sp1 = z2 + mid_h/2 + band/2;
    z1 = z_sp1 + band/2 + top_h/2;
    z_roof = z1 + top_h/2;
    z_chain = z_roof + roof_r;
    z_bracket = z_chain + chain_h;

    // Bottom tier: 2 cols, 1 row
    translate([0, 0, z3])
        tier(bot_w, bot_d, bot_h, 2, 1);

    // Tapered spacer bottom→mid
    translate([0, 0, z_sp2])
        spacer(mid_w, mid_d, bot_w, bot_d);

    // Middle tier: 2 cols, 2 rows
    translate([0, 0, z2])
        tier(mid_w, mid_d, mid_h, 2, 2);

    // Tapered spacer mid→top
    translate([0, 0, z_sp1])
        spacer(top_w, top_d, mid_w, mid_d);

    // Top tier: 3 cols, 2 rows
    translate([0, 0, z1])
        tier(top_w, top_d, top_h, 3, 2);

    // Barrel vault roof
    translate([0, 0, z_roof])
        barrel_roof(top_w, top_d);

    // Short chain
    translate([0, 0, z_chain])
        chain();

    // Bracket
    translate([0, 0, z_bracket])
        bracket();

    // Bottom cap
    color([0.15, 0.12, 0.1])
        translate([0, 0, z3 - bot_h/2 - 3])
            cube([bot_w * 0.5, bot_d * 0.5, 4], center=true);
}

lantern();
