<!DOCTYPE html>
<html>

<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>BingleFixIt - Phone, Computer Repair & IT Support</title>
    <meta name="description" content="Professional phone repair, computer repair, and IT support services">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="styles.css">

    <style>
        p {
            color: black;
        }
        .paragraph1 {
            font-size: 30px;
        }
        .paragraph2 {
            font-size: 13px;
        }
        .review {
            border: 1px solid #ccc;
            background: #f9f9f9;
            padding: 15px;
            margin: 15px 0;
        }
        .review p {
            margin-top: 5px;
        }
    </style>
</head>

<body>
    <script src="" async defer></script>
    
    <header class="banner">
        <h1>BingleFixIt</h1>
        <p>Phone Repair • Computer Repair • IT Support</p>
    </header>

    <div class="description-box">
        <h2>Professional Technology Repair & IT Services</h2>
        <p>
        We specialize in phone repairs, computer repairs, and IT support services. 
        From cracked screens and battery replacements to computer diagnostics and network troubleshooting,
        our certified technicians provide quality service with reliable solutions.
        Same-day repairs available for most common issues.
        </p>
    </div>


    <div class="services-section">
        <div class="service-box">
            <img src="https://placehold.co/400x400" alt="Phone Repair Badge" class="badge-icon">
            <p><strong>Phone Repair (Battery/Screen)</strong><br>Cracked screen or dead battery? We repair all major phone brands. Screen replacements, battery installations, and charging port fixes with quality parts.</p>
        </div>

        <div class="service-box">
            <img src="https://placehold.co/400x400" alt="Computer Repair Badge" class="badge-icon">
            <p><strong>Computer Repairs</strong><br>Laptop or desktop issues? We diagnose and fix hardware problems, software issues, virus removal, upgrades, and data recovery.</p>
        </div>

        <div class="service-box">
            <img src="https://placehold.co/400x400" alt="IT Support Badge" class="badge-icon">
            <p><strong>IT Support</strong><br>Network setup, troubleshooting, system maintenance, and technical support for homes and businesses. Remote and on-site assistance available.</p>
        </div>
    </div>

    <hr> 
    <h1>Leave a Review for BingleFixIt</h1>
    <p>Share your experience with our repair and IT support services</p>

    <form action="" method="POST">
        <label>Name:
            <input type="text" name="Name" required /></label><br />

        <label>Email:
            <input type="email" name="Email" required />
        </label><br />

        <label>Rating (out of 5):
            <input type="number" name="Rating" min="1" max="5" required />
        </label><br />

        <label>Tell us about your experience:<br />
            <textarea name="Comment" placeholder="How was your phone repair, computer repair, or IT support experience?" required></textarea></label><br />
        <input type="submit" name="Submit" value="Submit Review" />
    </form>

    <hr> 
    <?php
    // Enable error reporting for debugging
    error_reporting(E_ALL);
    ini_set('display_errors', 1);
    
    $env = parse_ini_file('.env');

    $host = $env['DB_HOST'];
    $port = $env['DB_PORT'];
    $db_name = $env['DB_NAME'];
    $username = $env['DB_USER'];
    $password = $env['DB_PASSWORD'];

    // Create connection with explicit port
    $conn = new mysqli($host, $username, $password, $db_name, $port);

    // Check connection
    if ($conn->connect_error)
    {
      die("<p style='color:red;'>Connection failed: ". $conn->connect_error . "</p>");
    }
    
    echo "<p style='color:green;'>Database connected successfully!</p>";

    // --- 1. HANDLE FORM SUBMISSION ---
    if (isset($_POST["Submit"]))
    {
      echo "<p style='color:blue;'>Form submitted, processing...</p>";
      
      $customer_name = $_POST["Name"];
      $email = $_POST["Email"];
      $rating = $_POST["Rating"];
      $review_content = $_POST["Comment"];
      
      echo "<p>Received: Name=$customer_name, Email=$email, Rating=$rating</p>";

      $stmt = $conn->prepare("INSERT INTO comments (customer_name, email, rating, review_content, created_at, updated_at) VALUES (?, ?, ?, ?, NOW(), NOW())");
      
      if (!$stmt) {
        echo "<p style='color:red;'>Prepare failed: " . $conn->error . "</p>";
      } else {
        $stmt->bind_param("ssis", $customer_name, $email, $rating, $review_content);

        if ($stmt->execute())
        {
          echo "<h2 style='color:green;'>✓ Thank you for reviewing BingleFixIt!</h2>";
        } 
        else
        {
          echo "<p style='color:red;'>Execute Error: ". $stmt->error . "</p>";
        }
        $stmt->close();
      }
    }

    // --- 2. DISPLAY EXISTING REVIEWS ---
    $sql = "SELECT customer_name, email, rating, review_content, created_at FROM comments ORDER BY created_at DESC";
    $result = $conn->query($sql);
    
    if (!$result) {
      echo "<p style='color:red;'>Query Error: " . $conn->error . "</p>";
    } else {
      echo "<p style='color:blue;'>Query executed. Found " . $result->num_rows . " reviews.</p>";
    }

    if ($result && $result->num_rows > 0)
    {
      echo"<h1>Customer Reviews:</h1><hr>";

      while( $row = $result->fetch_assoc())
      {
        echo "<div class='review'>
          <span><strong>Name:</strong> " . htmlspecialchars($row["customer_name"]) . "</span><br />
          <span><strong>Email:</strong> " . htmlspecialchars($row["email"]) . "</span><br />
          <span><strong>Rating:</strong> " . htmlspecialchars($row["rating"]) . "/5</span><br />
          <span><strong>Date:</strong> " . date("Y/m/d h:i A", strtotime($row["created_at"])) . "</span><br />
          <p><strong>Review:</strong> " . htmlspecialchars($row["review_content"]) . "</p>
        </div><hr>";
      }
    }
    else
    {
      echo "<p>No reviews yet!</p>";
    }

    $conn->close();
?>

</body>
</html>