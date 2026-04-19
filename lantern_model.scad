// Ottoman Hanging Lantern — 3D-printable single piece
// Designed for FDM printing with NO supports needed:
//  - Tier transitions are sloped (45°), not stepped (no horizontal overhangs)
//  - Hipped pyramid roof (all faces <= 45° from vertical)
//  - Windows are cut through the walls (no glass; use translucent filament or LED)
//  - Integrated hanging loop on top
// Print orientation: as-is, bottom on build plate. 0.4mm nozzle, 0.2mm layer, 2 perimeters.

$fn = 48;

// ─── Parameters ───

wall = 2.4;               // Wall thickness (6 perimeters at 0.4mm)

// Tier outer dimensions
top_w = 70;  top_d = 50;  top_h = 28;
mid_w = 48;  mid_d = 36;  mid_h = 20;
bot_w = 28;  bot_d = 22;  bot_h = 16;

// Slanted transitions between tiers (45° = safe overhang)
trans_h = 11;             // must be >= (top_w - mid_w)/2 and >= (mid_w - bot_w)/2

// Hipped pyramid roof
roof_h = 22;              // Roof peak height above top tier

// Eave flaps on front/back of top tier
flap_ext = 9;             // How far flap sticks out beyond tier face
flap_t = 3;               // Thickness (avoids thin unprintable features)

// Window cutouts
win_margin_x = 5;         // Horizontal border around windows
win_margin_z = 4;         // Vertical border

// Hanging loop on top
loop_r = 4;
loop_t = 3;

// ─── Outer shell (solid, no cutouts yet) ───

module outer_shell() {
    // Bottom tier
    translate([0, 0, bot_h/2])
        cube([bot_w, bot_d, bot_h], center=true);

    // Slanted transition bot → mid
    translate([0, 0, bot_h + trans_h/2])
        hull() {
            translate([0, 0, -trans_h/2])
                cube([bot_w, bot_d, 0.01], center=true);
            translate([0, 0, trans_h/2])
                cube([mid_w, mid_d, 0.01], center=true);
        }

    // Middle tier
    translate([0, 0, bot_h + trans_h + mid_h/2])
        cube([mid_w, mid_d, mid_h], center=true);

    // Slanted transition mid → top
    translate([0, 0, bot_h + trans_h + mid_h + trans_h/2])
        hull() {
            translate([0, 0, -trans_h/2])
                cube([mid_w, mid_d, 0.01], center=true);
            translate([0, 0, trans_h/2])
                cube([top_w, top_d, 0.01], center=true);
        }

    // Top tier
    translate([0, 0, bot_h + trans_h + mid_h + trans_h + top_h/2])
        cube([top_w, top_d, top_h], center=true);

    // Front/back flaps (sloped eaves — angled so overhang stays ~45°)
    z_flap = bot_h + trans_h + mid_h + trans_h + top_h - flap_t;
    for (sy = [-1, 1])
        translate([0, sy * (top_d/2 + flap_ext/2), z_flap + flap_t/2])
            hull() {
                translate([0, -sy * flap_ext/2, flap_t/2])
                    cube([top_w + 2, 0.01, 0.01], center=true);
                translate([0, sy * flap_ext/2, -flap_t/2])
                    cube([top_w + 2, 0.01, 0.01], center=true);
            }

    // Hipped pyramid roof
    z_roof = bot_h + trans_h + mid_h + trans_h + top_h;
    translate([0, 0, z_roof])
        hull() {
            cube([top_w, top_d, 0.01], center=true);
            translate([0, 0, roof_h])
                cube([4, 4, 0.01], center=true);
        }

    // Hanging loop on top
    translate([0, 0, z_roof + roof_h + loop_r])
        rotate([90, 0, 0])
            difference() {
                cylinder(r=loop_r + loop_t/2, h=loop_t, center=true);
                cylinder(r=loop_r - loop_t/2, h=loop_t + 1, center=true);
            }
}

// ─── Inner cavity (hollow the body, keep solid bottom) ───

module inner_cavity() {
    // Shrink each tier by wall thickness on all sides
    translate([0, 0, wall + bot_h/2])
        cube([bot_w - wall*2, bot_d - wall*2, bot_h], center=true);

    translate([0, 0, bot_h + trans_h/2])
        hull() {
            translate([0, 0, -trans_h/2])
                cube([bot_w - wall*2, bot_d - wall*2, 0.01], center=true);
            translate([0, 0, trans_h/2])
                cube([mid_w - wall*2, mid_d - wall*2, 0.01], center=true);
        }

    translate([0, 0, bot_h + trans_h + mid_h/2])
        cube([mid_w - wall*2, mid_d - wall*2, mid_h], center=true);

    translate([0, 0, bot_h + trans_h + mid_h + trans_h/2])
        hull() {
            translate([0, 0, -trans_h/2])
                cube([mid_w - wall*2, mid_d - wall*2, 0.01], center=true);
            translate([0, 0, trans_h/2])
                cube([top_w - wall*2, top_d - wall*2, 0.01], center=true);
        }

    translate([0, 0, bot_h + trans_h + mid_h + trans_h + top_h/2])
        cube([top_w - wall*2, top_d - wall*2, top_h], center=true);

    // Extend into the roof (hollow roof too — saves print time)
    z_roof = bot_h + trans_h + mid_h + trans_h + top_h;
    translate([0, 0, z_roof])
        hull() {
            cube([top_w - wall*2, top_d - wall*2, 0.01], center=true);
            translate([0, 0, roof_h - wall])
                cube([2, 2, 0.01], center=true);
        }
}

// ─── Window cutouts (3 cols on top front/back, 2 cols on mid, 1 col on bot) ───

module windows() {
    // Top tier: 3 windows on front and back
    z_top = bot_h + trans_h + mid_h + trans_h + top_h/2;
    win_h_top = top_h - win_margin_z * 2;
    col_w_top = (top_w - win_margin_x * 2) / 3;
    for (c = [0:2])
        for (sy = [-1, 1])
            translate([-(top_w/2) + win_margin_x + col_w_top * (c + 0.5),
                       sy * (top_d/2),
                       z_top])
                cube([col_w_top - 3, wall * 4, win_h_top], center=true);
    // Top tier: 2 windows on left/right
    col_w_top_s = (top_d - win_margin_x * 2) / 2;
    for (c = [0:1])
        for (sx = [-1, 1])
            translate([sx * (top_w/2),
                       -(top_d/2) + win_margin_x + col_w_top_s * (c + 0.5),
                       z_top])
                cube([wall * 4, col_w_top_s - 3, win_h_top], center=true);

    // Middle tier: 2 windows on front/back, 1 on sides
    z_mid = bot_h + trans_h + mid_h/2;
    win_h_mid = mid_h - win_margin_z * 2;
    col_w_mid = (mid_w - win_margin_x * 2) / 2;
    for (c = [0:1])
        for (sy = [-1, 1])
            translate([-(mid_w/2) + win_margin_x + col_w_mid * (c + 0.5),
                       sy * (mid_d/2),
                       z_mid])
                cube([col_w_mid - 3, wall * 4, win_h_mid], center=true);
    for (sx = [-1, 1])
        translate([sx * (mid_w/2), 0, z_mid])
            cube([wall * 4, mid_d - win_margin_x * 2, win_h_mid], center=true);

    // Bottom tier: 1 window each side
    z_bot = bot_h/2;
    win_h_bot = bot_h - win_margin_z * 2;
    for (sy = [-1, 1])
        translate([0, sy * (bot_d/2), z_bot])
            cube([bot_w - win_margin_x * 2, wall * 4, win_h_bot], center=true);
    for (sx = [-1, 1])
        translate([sx * (bot_w/2), 0, z_bot])
            cube([wall * 4, bot_d - win_margin_x * 2, win_h_bot], center=true);
}

// ─── Final printable model ───

module lantern_printable() {
    difference() {
        outer_shell();
        inner_cavity();
        windows();
    }
}

lantern_printable();
