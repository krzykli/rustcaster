use image::Pixel;
use rand::rngs::ThreadRng;
use rand::Rng;
use rayon::prelude::IntoParallelIterator;
use rayon::prelude::ParallelIterator;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use glam::vec3;
use glam::Vec3;

const EPSILON: f32 = 0.000001;

#[derive(Debug)]
struct RenderStats {
    rays: u32,
}

#[derive(Clone)]
struct Plane {
    normal: Vec3,
    distance: f32,
    material_id: u32,
}

impl Plane {
    fn intersect(&self, ray: &Ray) -> f32 {
        let denom = self.normal.dot(ray.direction);
        if denom.abs() < EPSILON {
            return f32::MAX;
        }

        (-self.distance - self.normal.dot(ray.origin)) / denom
    }
}
struct Ray {
    origin: Vec3,
    direction: Vec3,
}

#[derive(Clone)]
struct Sphere {
    pos: Vec3,
    radius: f32,
    material_id: u32,
}

impl Sphere {
    fn intersect(&self, ray: &Ray) -> f32 {
        let a = ray.direction.dot(ray.direction);
        let b = 2.0 * ray.direction.dot(ray.origin);
        let c = ray.origin.dot(ray.origin) - self.radius * self.radius;

        let denom = 2.0 * a;
        let root_term = ((b * b - 4.0 * a * c) as f32).sqrt();
        if root_term.abs() > EPSILON {
            let hit_far = (-b + root_term) / denom;
            let hit_near = (-b - root_term) / denom;

            if hit_near > 0.0 && hit_near < hit_far {
                return hit_near;
            }
        }

        f32::MAX
    }
}

#[derive(Default, Clone)]
struct Material {
    label: String,
    shininess: f32,
    emit_color: Vec3,
    reflect_color: Vec3,
}

#[derive(Clone)]
struct Camera {
    pos: Vec3,
}

#[derive(Clone)]
struct World {
    materials: Vec<Material>,
    planes: Vec<Plane>,
    spheres: Vec<Sphere>,
    camera: Camera,
}

fn init_materials() -> Vec<Material> {
    let mut materials = vec![];
    materials.push(Material {
        label: String::from("sky"),
        emit_color: Vec3 {
            x: 0.3,
            y: 0.4,
            z: 0.6,
        },
        ..Default::default()
    });

    materials.push(Material {
        label: String::from("plane"),
        shininess: 0.0,
        reflect_color: Vec3 {
            x: 0.3,
            y: 0.3,
            z: 0.3,
        },
        ..Default::default()
    });

    materials.push(Material {
        label: String::from("sphere1"),
        shininess: 0.0,
        reflect_color: Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        ..Default::default()
    });

    materials.push(Material {
        label: String::from("sphere2"),
        shininess: 0.5,
        reflect_color: Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
        ..Default::default()
    });

    materials.push(Material {
        label: String::from("sphere3"),
        shininess: 0.95,
        emit_color: Vec3 {
            x: 0.0,
            y: 0.0,
            z: 100.0,
        },
        reflect_color: Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        ..Default::default()
    });

    materials.push(Material {
        label: String::from("sphere4"),
        shininess: 0.8,
        emit_color: Vec3 {
            x: 0.1,
            y: 0.1,
            z: 0.1,
        },
        reflect_color: Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        ..Default::default()
    });

    materials
}

fn init_planes() -> Vec<Plane> {
    let mut planes = vec![];

    planes.push(Plane {
        material_id: 1,
        normal: vec3(0.0, 0.0, 1.0),
        distance: 0.0,
    });

    planes
}

fn init_spheres() -> Vec<Sphere> {
    let mut spheres = vec![];

    spheres.push(Sphere {
        material_id: 2,
        pos: vec3(-3.0, 0.0, 2.0),
        radius: 1.0,
    });

    spheres.push(Sphere {
        material_id: 3,
        pos: vec3(0.0, -3.0, 1.0),
        radius: 1.0,
    });

    spheres.push(Sphere {
        material_id: 4,
        pos: vec3(3.0, 0.0, 0.0),
        radius: 1.0,
    });

    spheres.push(Sphere {
        material_id: 5,
        pos: vec3(5.0, -3.5, 1.5),
        radius: 2.0,
    });

    spheres
}

fn linear_to_srgb(c: &Vec3) -> Vec3 {
    vec3(255.0 * c.x.sqrt(), 255.0 * c.y.sqrt(), 255.0 * c.z.sqrt())
}

fn raycast(world: &World, ray: &Ray, bounces: u32) -> Vec3 {
    let mut result = vec3(0.0, 0.0, 0.0);
    let mut attenuation = vec3(1.0, 1.0, 1.0);

    let mut hit_dist = f32::MAX;
    let min_hit_distance = EPSILON;
    let mut next_origin = ray.origin;

    let mut ray = Ray {
        origin: ray.origin,
        direction: ray.direction,
    };

    for _ in 0..bounces {
        let mut hit_mat_idx = 0;
        let mut next_normal = vec3(0.0, 0.0, 0.0);

        for plane in &world.planes {
            let this_dist = plane.intersect(&ray);

            if (this_dist > min_hit_distance) && (this_dist < hit_dist) {
                hit_dist = this_dist;
                hit_mat_idx = plane.material_id;
                next_normal = plane.normal;
            }
        }

        for sphere in &world.spheres {
            let sphere_ray_origin = ray.origin - sphere.pos;

            let this_dist = sphere.intersect(&Ray {
                origin: sphere_ray_origin,
                direction: ray.direction,
            });

            if (this_dist > min_hit_distance) && (this_dist < hit_dist) {
                hit_dist = this_dist;
                hit_mat_idx = sphere.material_id;
                next_normal = hit_dist * ray.direction + sphere_ray_origin;
            }
        }

        if hit_mat_idx != 0 {
            let mat = &world.materials[hit_mat_idx as usize];

            let mut rng = rand::thread_rng();
            let r1 = -1.0 + 2.0 * rng.gen::<f32>();
            let r2 = -1.0 + 2.0 * rng.gen::<f32>();
            let r3 = -1.0 + 2.0 * rng.gen::<f32>();

            next_origin += hit_dist * ray.direction;

            let direct_reflection =
                (ray.direction - 2.0 * ray.direction.dot(next_normal) * next_normal).normalize();
            let random_reflection = (next_normal + vec3(r1, r2, r3)).normalize();

            let next_dir = (random_reflection.lerp(direct_reflection, mat.shininess)).normalize();

            result += attenuation * mat.emit_color;

            let cos_attentuation = 1.0;
            // let mut cos_attentuation = (-ray.direction).dot(next_normal);
            // if cos_attentuation < 0.0 {
            //     cos_attentuation = 0.0;
            // }
            attenuation = attenuation * cos_attentuation * mat.reflect_color;

            ray = Ray {
                origin: next_origin,
                direction: next_dir,
            };
        } else {
            let mat = &world.materials[0];
            result += attenuation * mat.emit_color;
            break;
        }
    }

    return vec3(result.x as f32, result.y as f32, result.z as f32);
}

fn rand_bilateral(rng: &mut ThreadRng) -> f32 {
    -1.0 + 2.0 * rng.gen::<f32>()
}

#[derive(Copy, Clone)]
struct Bucket {
    x_min: u32,
    y_min: u32,
    size: u32,
}

fn render_bucket(
    buffer: Arc<Mutex<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>>>,
    bucket: &Bucket,
    world: &World,
    render_settings: &RenderSettings,
) {
    let half_px_width = 0.5 / render_settings.output_width as f32;
    let half_px_height = 0.5 / render_settings.output_height as f32;

    let output_width = render_settings.output_width;
    let output_height = render_settings.output_height;

    let contrib = 1.0 / render_settings.samples as f32;
    let mut casted_rays = 0;
    let ray_origin = world.camera.pos;
    let mut rng = rand::thread_rng();

    let camera_z = (world.camera.pos - vec3(0.0, 0.0, 0.0)).normalize();
    let camera_x = (camera_z.cross(vec3(0.0, 0.0, 1.0))).normalize();
    let camera_y = (camera_z.cross(camera_x)).normalize();

    let film_dist = 1.0;
    let film_center = world.camera.pos - film_dist * camera_z;
    let mut film_width = 1.0;
    let mut film_height = 1.0;

    if output_width > output_height {
        film_height = film_width * output_height as f32 / output_width as f32;
    } else if output_height > output_width {
        film_width = film_height * output_width as f32 / output_height as f32;
    };

    let mut sub_image = image::ImageBuffer::new(bucket.size, bucket.size);

    for y in 0..bucket.size {
        for x in 0..bucket.size {
            let pixel = sub_image.get_pixel_mut(x, y);

            let image_coord_x = x + bucket.x_min;
            let image_coord_y = y + bucket.y_min;

            let mut color = vec3(0.0, 0.0, 0.0);
            let film_x = -1.0 + 2.0 * (image_coord_x as f32 / render_settings.output_width as f32);
            let film_y = -1.0 + 2.0 * (image_coord_y as f32 / render_settings.output_height as f32);

            for _ in 0..render_settings.samples {
                let offset_x = film_x + rand_bilateral(&mut rng) * half_px_width;
                let offset_y = film_y + rand_bilateral(&mut rng) * half_px_height;
                let film_p = film_center
                    + offset_x * film_width * 0.5 * camera_x
                    + offset_y * film_height * 0.5 * camera_y;

                let ray_direction = (film_p - world.camera.pos).normalize();

                let ray = Ray {
                    origin: ray_origin,
                    direction: ray_direction,
                };

                color += contrib * raycast(&world, &ray, render_settings.bounces);
                casted_rays += 1;
            }
            color = linear_to_srgb(&color);

            *pixel = image::Rgb([(color.x) as u8, (color.y) as u8, (color.z) as u8]);
        }
    }

    let mut buffer = buffer.lock().unwrap();
    for y in 0..bucket.size {
        for x in 0..bucket.size {
            buffer.put_pixel(
                (x + bucket.x_min).min(render_settings.output_width - 1),
                (y + bucket.y_min).min(render_settings.output_height - 1),
                *sub_image.get_pixel(x, y),
            );
        }
    }
}

#[derive(Copy, Clone)]
struct RenderSettings {
    output_width: u32,
    output_height: u32,
    samples: u32,
    bounces: u32,
}

fn main() {
    // render settings
    //
    let output_width = 1280;
    let output_height = 720;

    let render_settings = RenderSettings {
        output_width,
        output_height,
        samples: 8,
        bounces: 4,
    };

    let buffer =
        image::ImageBuffer::new(render_settings.output_width, render_settings.output_height);
    let file_path = "render.png";

    // camera
    let camera = Camera {
        pos: vec3(0.0, 10.0, 1.0),
    };

    // world
    let world = World {
        materials: init_materials(),
        planes: init_planes(),
        spheres: init_spheres(),
        camera,
    };

    let bucket_size = 32;

    let mut buckets = vec![];

    let mut buckets_x_count = output_width / bucket_size;
    let mut buckets_y_count = output_height / bucket_size;
    if output_width % bucket_size != 0 {
        buckets_x_count += 1
    }
    if output_height % bucket_size != 0 {
        buckets_y_count += 1
    }
    let buckets_x_count = buckets_x_count;
    let buckets_y_count = buckets_y_count;

    for y in 0..buckets_y_count {
        for x in 0..buckets_x_count {
            buckets.push(Bucket {
                x_min: x * bucket_size,
                y_min: y * bucket_size,
                size: bucket_size,
            })
        }
    }
    let buckets = buckets;

    let cores = num_cpus::get_physical();
    println!("Using {} cores", cores);

    let start = Instant::now();

    let mut_buffer = Arc::new(Mutex::new(buffer));
    buckets.into_par_iter()
        .for_each(|b| render_bucket(mut_buffer.clone(), &b, &world, &render_settings));

    // let mut thread_handles = vec![];

    // let mut_buffer = Arc::new(Mutex::new(buffer));
    // for bucket in buckets {
    //     let arc_buf = Arc::clone(&mut_buffer);
    //     let my_buc = bucket.clone();
    //     let my_world = world.clone();
    //     let handle = thread::spawn(move || {
    //         render_bucket(arc_buf, &my_buc, &my_world, &render_settings);
    //     });
    //     thread_handles.push(handle);
    // }

    // for handle in thread_handles {
    //     handle.join().unwrap();
    // }

    let time_elapsed = start.elapsed();
    println!("{:?}", time_elapsed);

    // println!(
    //     "{:.8} ms/bounce",
    //     time_elapsed.as_millis() as f32 / casted_rays as f32
    // );
    println!("{:#?}", time_elapsed);
    mut_buffer.lock().unwrap().save(file_path).unwrap();
    println!("file saved to {}", file_path);
}
