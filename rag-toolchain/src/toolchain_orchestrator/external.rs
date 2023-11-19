use std::io::Error;

pub trait Source {
    fn read_from_source(&self) -> Result<Vec<String>, Error>;
}

pub trait Destination {
    fn write_to_dest(&self, text: String) -> Result<(), Error>;
}
