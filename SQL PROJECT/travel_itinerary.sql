use `travel_itinerary`;

CREATE TABLE Users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100),
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(20)
);

CREATE TABLE Trips (
    trip_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    trip_name VARCHAR(100),
    start_date DATE,
    end_date DATE,
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);


CREATE TABLE destinations (
    destination_id INT AUTO_INCREMENT PRIMARY KEY,
    destination_name VARCHAR(100) NOT NULL,
    country VARCHAR(100) NOT NULL,
    description TEXT
);

CREATE TABLE restaurants (    
    restaurant_id INT AUTO_INCREMENT PRIMARY KEY,
    destination_id INT,
    name VARCHAR(100),
    type_of_cuisine VARCHAR(100),
    rating DECIMAL(2,1),
    average_cost_per_meal DECIMAL(6,2),
    FOREIGN KEY (destination_id) REFERENCES destinations(destination_id)
);

CREATE TABLE activities (
    activity_id INT AUTO_INCREMENT PRIMARY KEY,
    destination_id INT,
    name VARCHAR(100),
    type VARCHAR(50),
    description TEXT,
    cost DECIMAL(6,2),
    FOREIGN KEY (destination_id) REFERENCES destinations(destination_id)
);

CREATE TABLE accommodations (
    accommodation_id INT AUTO_INCREMENT PRIMARY KEY,
    destination_id INT,
    name VARCHAR(100),
    type VARCHAR(50),
    price_per_night DECIMAL(6,2),
    rating DECIMAL(2,1),
    FOREIGN KEY (destination_id) REFERENCES destinations(destination_id)
);

CREATE TABLE transportation (
    transport_id INT AUTO_INCREMENT PRIMARY KEY,
    itinerary_id INT,
    from_destination_id INT,
    to_destination_id INT,
    transport_mode VARCHAR(50),
    departure_time DATETIME,
    arrival_time DATETIME,
    cost DECIMAL(6,2),
    FOREIGN KEY (from_destination_id) REFERENCES destinations(destination_id),
    FOREIGN KEY (to_destination_id) REFERENCES destinations(destination_id)
);

CREATE TABLE itineraries (
    itinerary_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    start_date DATE,
    end_date DATE,
    total_cost DECIMAL(10,2),
    description TEXT
);

CREATE TABLE Expenses (
    expense_id INT PRIMARY KEY AUTO_INCREMENT,
    trip_id INT,
    category VARCHAR(50),
    amount DECIMAL(10, 2),
    date DATE,
    FOREIGN KEY (trip_id) REFERENCES Trips(trip_id)
);

INSERT INTO Users (name, email, phone)
VALUES 
('John Doe', 'john@example.com', '123-456-7890'),
('Jane Smith', 'jane@example.com', '987-654-3210'),
('Emily Johnson', 'emily@example.com', '555-123-4567'),
('Michael Brown', 'michael@example.com', '901-234-5678'),
('Sarah Lee', 'sarah@example.com', '111-222-3333'),
('David White', 'david@example.com', '444-555-6666'),
('Olivia Martin', 'olivia@example.com', '777-888-9999'),
('Kevin Davis', 'kevin@example.com', '333-444-5555'),
('Rebecca Garcia', 'rebecca@example.com', '666-777-8888'),
('Christopher Harris', 'christopher@example.com', '999-000-1111');

INSERT INTO Trips (user_id, trip_name, start_date, end_date)
VALUES
(1, 'Europe Adventure', '2024-06-01', '2024-06-20'),
(2, 'Asia Discovery', '2024-07-10', '2024-07-25'),
(3, 'American Road Trip', '2024-08-01', '2024-08-15'),
(1, 'Caribbean Cruise', '2024-09-05', '2024-09-19'),
(4, 'African Safari', '2024-10-01', '2024-10-12'),
(5, 'South American Exploration', '2025-01-05', '2025-01-22'),
(2, 'Australian Odyssey', '2025-02-20', '2025-03-05'),
(6, 'Middle Eastern Journey', '2025-04-01', '2025-04-15'),
(3, 'Alaskan Adventure', '2025-05-10', '2025-05-25'),
(7, 'Indian Himalayas Trek', '2025-06-01', '2025-06-18'),
(8, 'New Zealand Discovery', '2025-07-05', '2025-07-22');

INSERT INTO destinations (destination_name, country, description)
VALUES
('Eiffel Tower', 'France', 'Iconic iron lattice tower in Paris, symbolizing French culture'),
('Great Barrier Reef', 'Australia', 'Worlds largest coral reef system, home to diverse marine life'),
('Taj Mahal', 'India', 'Stunning white marble mausoleum in Agra, representing love and beauty'),
('Grand Canyon', 'USA', 'Breathtaking natural wonder in Arizona, carved by the Colorado River'),
('Great Wall of China', 'China', 'Ancient series of fortifications, showcasing engineering and history'),
('Machu Picchu', 'Peru', 'Mysterious Inca citadel, nestled in the Andes mountains'),
('Pyramids of Giza', 'Egypt', 'Ancient wonders, last remaining of the Seven Wonders of the Ancient World'),
('Sydney Opera House', 'Australia', 'Sculptural performing arts center, iconic symbol of Sydney'),
('Burj Khalifa', 'UAE', 'Tallest building in the world, modern architectural marvel'),
('Chichén Itzá', 'Mexico', 'Ancient Mayan city, rich in history and cultural significance');

INSERT INTO restaurants (destination_id, name, type_of_cuisine, rating, average_cost_per_meal)
VALUES
(1, 'Le Jules Verne', 'French', 4.8, 120.00),
(2, 'The Fish House', 'Seafood', 4.5, 40.00),
(3, 'Taj Mahal Restaurant', 'Indian', 4.7, 30.00),
(4, 'El Tovar Dining Room', 'Southwestern', 4.6, 50.00),
(5, 'Jade Garden', 'Chinese', 4.4, 25.00),
(6, 'Indio Feliz', 'Peruvian', 4.9, 35.00),
(7, 'Koshari Restaurant', 'Middle Eastern', 4.5, 20.00),
(8, 'Quay Restaurant', 'Australian', 4.8, 60.00),
(9, 'At.mosphere', 'International', 4.7, 100.00),
(10, 'Hacienda Tres Rios', 'Mexican', 4.6, 40.00);

INSERT INTO activities (destination_id, name, type, description, cost)
VALUES
(1, 'Eiffel Tower Summit Tour', 'Guided Tour', 'Ascend to the top for breathtaking views', 32.00),
(2, 'Great Barrier Reef Snorkeling', 'Water Sports', 'Explore vibrant coral and marine life', 85.00),
(3, 'Taj Mahal Sunrise Tour', 'Guided Tour', 'Witness the monument’s beauty at dawn', 25.00),
(4, 'Grand Canyon Helicopter Tour', 'Air Tour', 'Soar above the Colorado River and canyon', 150.00),
(5, 'Great Wall of China Hike', 'Hiking', 'Explore the ancient fortifications', 40.00),
(6, 'Machu Picchu Inca Trail', 'Hiking', 'Follow the historic Inca path', 80.00),
(7, 'Pyramids of Giza Camel Ride', 'Adventure', 'Ride through the desert to the pyramids', 50.00),
(8, 'Sydney Harbour Cruise', 'Water Tour', 'Admire iconic landmarks from the water', 60.00),
(9, 'Burj Khalifa Skydiving', 'Adventure', 'Experience the thrill of skydiving', 200.00),
(10, 'Chichén Itzá Mayan Ruins Tour', 'Guided Tour', 'Uncover the secrets of the ancient city', 30.00);

INSERT INTO accommodations (destination_id, name, type, price_per_night, rating)
VALUES
(1, 'Shangri-La Hotel Paris', 'Luxury Hotel', 450.00, 4.9),
(2, 'The Reef Hotel', 'Beach Resort', 250.00, 4.5),
(3, 'Taj Mahal Palace Mumbai', 'Luxury Hotel', 300.00, 4.8),
(4, 'El Tovar Hotel', 'Historic Hotel', 200.00, 4.6),
(5, 'The Great Wall Hotel Beijing', 'Business Hotel', 150.00, 4.3),
(6, 'Inkaterra Machu Picchu Pueblo Hotel', 'Boutique Hotel', 180.00, 4.7),
(7, 'Mena House Hotel', 'Luxury Hotel', 220.00, 4.5),
(8, 'Shangri-La Hotel Sydney', 'Luxury Hotel', 280.00, 4.8),
(9, 'Armani Hotel Dubai', 'Luxury Hotel', 380.00, 4.9),
(10, 'Hotel Mayaland', 'Resort', 120.00, 4.4);

INSERT INTO transportation (itinerary_id, from_destination_id, to_destination_id, transport_mode, departure_time, arrival_time, cost)
VALUES
(1, 1, 2, 'Flight', '2024-06-01 08:00:00', '2024-06-01 17:00:00', 500.00),
(1, 2, 3, 'Flight', '2024-06-03 09:00:00', '2024-06-03 14:00:00', 400.00),
(2, 4, 5, 'Train', '2024-07-10 07:00:00', '2024-07-10 12:00:00', 200.00),
(2, 5, 6, 'Bus', '2024-07-12 08:00:00', '2024-07-12 16:00:00', 150.00),
(3, 7, 8, 'Flight', '2024-08-15 10:00:00', '2024-08-15 19:00:00', 600.00),
(3, 8, 9, 'Taxi', '2024-08-17 09:00:00', '2024-08-17 10:00:00', 50.00),
(4, 10, 1, 'Flight', '2024-09-01 08:00:00', '2024-09-01 15:00:00', 450.00),
(4, 1, 4, 'Train', '2024-09-03 07:00:00', '2024-09-03 13:00:00', 250.00),
(5, 6, 7, 'Bus', '2024-10-10 09:00:00', '2024-10-10 17:00:00', 120.00),
(5, 7, 10, 'Flight', '2024-10-12 10:00:00', '2024-10-12 18:00:00', 550.00);

INSERT INTO itineraries (user_id, start_date, end_date, total_cost, description)
VALUES
(1, '2024-06-01', '2024-06-20', 3000.00, 'Europe Adventure'),
(2, '2024-07-10', '2024-07-25', 2500.00, 'Asia Discovery'),
(3, '2024-08-15', '2024-08-29', 4000.00, 'Australia and New Zealand Explorer'),
(4, '2024-09-01', '2024-09-15', 3500.00, 'South America Odyssey'),
(5, '2024-10-10', '2024-10-24', 2800.00, 'Middle East and Africa Adventure'),
(6, '2024-11-01', '2024-11-14', 3200.00, 'North America Road Trip'),
(7, '2024-12-01', '2024-12-15', 3800.00, 'Caribbean Cruise'),
(8, '2025-01-05', '2025-01-19', 4200.00, 'Southeast Asia Expedition'),
(9, '2025-02-10', '2025-02-24', 4500.00, 'South Pacific Islands Getaway'),
(10, '2025-03-01', '2025-03-14', 4800.00, 'European River Cruise');

INSERT INTO Expenses (trip_id, category, amount, date)
VALUES
(1, 'Transportation', 500.00, '2024-06-01'),
(1, 'Accommodation', 800.00, '2024-06-02'),
(2, 'Food', 200.00, '2024-07-10'),
(2, 'Activities', 300.00, '2024-07-12'),
(3, 'Flight', 1200.00, '2024-08-15'),
(3, 'Travel Insurance', 150.00, '2024-08-16'),
(4, 'Entry Fees', 100.00, '2024-09-01'),
(4, 'Guided Tour', 250.00, '2024-09-03'),
(5, 'Souvenirs', 50.00, '2024-10-10'),
(5, 'Miscellaneous', 200.00, '2024-10-12');

show tables;

describe users;
describe trips;

-- Get all trips booked by a specific user --
SELECT *
FROM Trips
WHERE user_id = 1;

 -- Find all destinations visited by a user --

SELECT D.destination_name
FROM Users U
JOIN Trips T ON U.user_id = T.user_id
JOIN Destinations D ON T.trip_id = D.trip_id
WHERE U.user_id = 1;

-- Find all activities booked for a specific trip

SELECT A.activity_name
FROM Trips T
JOIN Trip_Activities TA ON T.trip_id = TA.trip_id
JOIN Activities A ON TA.activity_id = A.activity_id
WHERE T.trip_id = 5;

-- Get total cost of a trip --
SELECT SUM(A.cost)
FROM Trips T
JOIN Trip_Activities TA ON T.trip_id = TA.trip_id
JOIN Activities A ON TA.activity_id = A.activity_id
WHERE T.trip_id = 5;

 -- Get trip itinerary for a specific trip ID

SELECT D.destination_name, A.activity_name, TA.start_date, TA.end_date
FROM Trips T
JOIN Destinations D ON T.trip_id = D.trip_id
JOIN Trip_Activities TA ON T.trip_id = TA.trip_id
JOIN Activities A ON TA.activity_id = A.activity_id
WHERE T.trip_id = 5
ORDER BY TA.start_date;

--  Find top 3 most popular destinations

SELECT D.destination_name, COUNT(T.trip_id) AS num_trips
FROM Destinations D
JOIN Trips T ON D.trip_id = T.trip_id
GROUP BY D.destination_name
ORDER BY num_trips DESC
LIMIT 3;

-- Get average trip cost per destination

SELECT D.destination_name, 
AVG(T.trip_cost) AS avg_trip_cost
FROM Trips T
JOIN Destinations D ON T.trip_id = D.trip_id
GROUP BY D.destination_name
ORDER BY avg_trip_cost DESC;

-- Get trip details with activity names
SELECT 
  T.trip_id, 
  T.trip_name, 
  A.activity_name, 
  TA.start_date, 
  TA.end_date
FROM Trips T
JOIN Trip_Activities TA ON T.trip_id = TA.trip_id
JOIN Activities A ON TA.activity_id = A.activity_id
ORDER BY 
  T.trip_id, 
  TA.start_date;
  
 -- Find destinations with no trips booked
SELECT D.destination_name
FROM Destinations D
LEFT JOIN Trips T ON D.trip_id = T.trip_id
WHERE T.trip_id IS NULL; 

describe destinations;
SELECT 
  D.destination_name
FROM 
  Users U
JOIN 
  Trips T ON U.user_id = T.user_id
JOIN 
  Destinations D ON T.destination_id = D.destination_id
WHERE 
  U.user_id = 1
LIMIT 0, 1000;
