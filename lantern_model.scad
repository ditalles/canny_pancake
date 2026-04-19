// Ottoman/Turkish Hanging Lantern — 4-sided, 3-tier, barrel-vault top
// White glass faces with dark metal mullion grids
// Render: F5 preview, F6 full render, export STL for 3D printing

$fn = 50;

// ─── Parameters ───

// Frame
frame = 2.5;        // Metal frame bar width

// Tier sizes [width, depth, height] — top is largest
top_w = 72;   top_d = 52;   top_h = 48;
mid_w = 56;   mid_d = 42;   mid_h = 42;
bot_w = 36;   bot_d = 28;   bot_h = 22;

band = 5;           // Spacer band height between tiers
roof_r = 28;        // Barrel vault radius

chain_h = 35;
bracket_reach = 110;

// ─── Helper: one glass face with mullion grid ───

module glass_face(w, h, cols, rows) {
    // White glass background
    color([0.92, 0.9, 0.85])
        cube([w, 1.2, h], center=true);

    // Outer border frame
    color([0.12, 0.1, 0.08]) {
        // Top & bottom bars
        translate([0, 0, h/2])  cube([w + 1, frame, frame], center=true);
        translate([0, 0, -h/2]) cube([w + 1, frame, frame], center=true);
        // Left & right bars
        translate([w/2, 0, 0])  cube([frame, frame, h + 1], center=true);
        translate([-w/2, 0, 0]) cube([frame, frame, h + 1], center=true);
    }

    // Interior mullion grid
    color([0.12, 0.1, 0.08]) {
        // Vertical dividers
        for (c = [1:cols-1]) {
            xpos = -w/2 + c * (w / cols);
            translate([xpos, 0, 0])
                cube([frame * 0.7, frame, h], center=true);
        }
        // Horizontal dividers
        for (r = [1:rows-1]) {
            zpos = -h/2 + r * (h / rows);
            translate([0, 0, zpos])
                cube([w, frame, frame * 0.7], center=true);
        }
    }
}

// ─── One complete tier ───

module tier(w, d, h, cols, rows) {
    // 4 glass faces
    // Front
    translate([0, d/2, 0])
        glass_face(w - frame*2, h - frame*2, cols, rows);
    // Back
    translate([0, -d/2, 0])
        glass_face(w - frame*2, h - frame*2, cols, rows);
    // Right
    translate([w/2, 0, 0])
        rotate([0, 0, 90])
            glass_face(d - frame*2, h - frame*2, cols, rows);
    // Left
    translate([-w/2, 0, 0])
        rotate([0, 0, 90])
            glass_face(d - frame*2, h - frame*2, cols, rows);

    // 4 vertical corner posts
    color([0.12, 0.1, 0.08])
    for (sx = [-1, 1])
        for (sy = [-1, 1])
            translate([sx * w/2, sy * d/2, 0])
                cube([frame, frame, h], center=true);

    // Top and bottom cap plates
    color([0.15, 0.12, 0.1]) {
        translate([0, 0, h/2])
            cube([w + 2, d + 2, frame], center=true);
        translate([0, 0, -h/2])
            cube([w + 2, d + 2, frame], center=true);
    }
}

// ─── Spacer band between tiers ───

module spacer(w, d) {
    color([0.15, 0.12, 0.1])
        cube([w + 3, d + 3, band], center=true);
}

// ─── Barrel vault roof ───

module barrel_roof(w, d) {
    color([0.15, 0.12, 0.1]) {
        // Base plate
        translate([0, 0, -1])
            cube([w + 4, d + 4, frame], center=true);

        // Half-cylinder vault
        difference() {
            // Outer vault
            rotate([0, 90, 0])
                cylinder(r=roof_r, h=w + 4, center=true);
            // Hollow inside
            rotate([0, 90, 0])
                cylinder(r=roof_r - frame, h=w + 6, center=true);
            // Cut away bottom half
            translate([0, 0, -roof_r])
                cube([w + 10, roof_r * 3, roof_r * 2], center=true);
        }

        // Ridge bar on top
        translate([0, 0, roof_r - 1])
            cube([w + 6, frame, frame], center=true);

        // End caps (solid half-circles)
        for (sx = [-1, 1])
            translate([sx * (w/2 + 1.5), 0, 0])
            rotate([0, 90, 0])
                difference() {
                    cylinder(r=roof_r, h=frame, center=true);
                    translate([0, 0, 0])
                        cylinder(r=roof_r - frame * 2, h=frame + 1, center=true);
                    translate([0, 0, -roof_r])
                        cube([roof_r * 3, roof_r * 3, roof_r * 2], center=true);
                }

        // End cap glass fill
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

    // Front/back glass arches
    for (sy = [-1, 1])
        color([0.92, 0.9, 0.85])
        translate([0, sy * (d/2), 0])
            rotate([0, 90, 0])
                difference() {
                    cylinder(r=roof_r - frame - 0.5, h=w - frame*2, center=true);
                    cylinder(r=roof_r - frame - 1.5, h=w - frame*2 + 1, center=true);
                    translate([0, 0, -roof_r])
                        cube([roof_r * 3, roof_r * 3, roof_r * 2], center=true);
                }
}

// ─── Chain ───

module chain() {
    color([0.2, 0.17, 0.14])
    for (i = [0:floor(chain_h/7)-1]) {
        translate([0, 0, i * 7]) {
            if (i % 2 == 0) {
                difference() {
                    cube([2.5, 5, 7], center=true);
                    cube([1, 3, 5], center=true);
                }
            } else {
                difference() {
                    cube([5, 2.5, 7], center=true);
                    cube([3, 1, 5], center=true);
                }
            }
        }
    }
}

// ─── Wall bracket with scroll ───

module bracket() {
    color([0.12, 0.1, 0.08]) {
        // Wall plate
        translate([-bracket_reach, 0, 0])
            cube([8, 35, 55], center=true);

        // Main arm
        hull() {
            translate([-bracket_reach + 8, 0, 15])
                cube([4, 5, 5], center=true);
            translate([0, 0, 0])
                cube([4, 5, 5], center=true);
        }

        // Support strut
        hull() {
            translate([-bracket_reach + 8, 0, -10])
                cube([4, 5, 5], center=true);
            translate([-25, 0, 10])
                cube([4, 5, 5], center=true);
        }

        // Decorative scroll circle
        translate([-40, 0, 8])
        rotate([90, 0, 0])
            difference() {
                cylinder(r=14, h=4, center=true);
                cylinder(r=10, h=5, center=true);
            }

        // Hook ring at end
        translate([3, 0, -5])
        rotate([90, 0, 0])
            difference() {
                cylinder(r=7, h=4, center=true);
                cylinder(r=5, h=5, center=true);
                translate([0, -7, 0])
                    cube([16, 8, 6], center=true);
            }
    }
}

// ─── Full assembly ───

module lantern() {
    // Stack from bottom up
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

    // Spacer
    translate([0, 0, z_sp2])
        spacer(bot_w, bot_d);

    // Middle tier: 2 cols, 2 rows
    translate([0, 0, z2])
        tier(mid_w, mid_d, mid_h, 2, 2);

    // Spacer
    translate([0, 0, z_sp1])
        spacer(mid_w, mid_d);

    // Top tier: 3 cols, 2 rows
    translate([0, 0, z1])
        tier(top_w, top_d, top_h, 3, 2);

    // Barrel vault roof
    translate([0, 0, z_roof])
        barrel_roof(top_w, top_d);

    // Chain
    translate([0, 0, z_chain])
        chain();

    // Wall bracket
    translate([0, 0, z_bracket])
        bracket();

    // Bottom drop cap
    color([0.15, 0.12, 0.1])
        translate([0, 0, z3 - bot_h/2 - 4])
            cube([bot_w * 0.6, bot_d * 0.6, 5], center=true);
}

lantern();
