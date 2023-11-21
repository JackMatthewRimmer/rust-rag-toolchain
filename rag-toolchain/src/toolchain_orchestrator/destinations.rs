use dotenv::dotenv;

pub struct PgVector {
    username: String,
    password: String,
    host: String,
    db_name: String,
}

impl PgVector {
    pub fn new() -> PgVector {
        dotenv().ok();
        PgVector {
            username: dotenv::var("POSTGRES_USERNAME").expect("PG_USERNAME must be set"),
            password: dotenv::var("POSTGRES_PASSWORD").expect("PG_PASSWORD must be set"),
            host: dotenv::var("POSTGRES_HOST").expect("PG_HOST must be set"),
            db_name: dotenv::var("POSTGRES_DB_NAME").expect("PG_DB_NAME must be set"),
        }
    }
}
