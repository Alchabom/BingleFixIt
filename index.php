<!DOCTYPE html>
<!--[if lt IE 7]>      <html class="no-js lt-ie9 lt-ie8 lt-ie7"> <![endif]-->
<!--[if IE 7]>         <html class="no-js lt-ie9 lt-ie8"> <![endif]-->
<!--[if IE 8]>         <html class="no-js lt-ie9"> <![endif]-->
<!--[if gt IE 8]>      <html class="no-js"> <!--<![endif]-->
<html>
    <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <title></title>
        <meta name="description" content="">
        <meta name="viewport" content="width=device-width, initial-scale=1">
         <link rel="stylesheet" href="styles.css">
    </head>

    <body>
        <!--[if lt IE 7]>
            <p class="browsehappy">You are using an <strong>outdated</strong> browser. Please <a href="#">upgrade your browser</a> to improve your experience.</p>
        <![endif]-->
        <script src="" async defer></script>
        <header class = "banner">
          
        </header>

        <div class = "description-box">
          Lorem ipsum dolor sit amet, 
          consectetur adipiscing elit. 
          Aliquam cursus lectus non tellus venenatis, vel bibendum nisi faucibus. 
          Proin vehicula dignissim magna a posuere. Donec tempus ex id arcu tempor pulvinar. 
          Phasellus mattis ullamcorper elit semper viverra. 
        </div>


        <div class = "services-section">
          <div class = "service-box">
            <img src="https://placehold.co/400x400" alt="Service 1 Badge" class="badge-icon">
            <p>Service  1 Description</p>
          </div>


        <div class = "service-box">
          <img src="https://placehold.co/400x400" alt="Service 2 Badge" class="badge-icon">
          <p>Service  2 Description</p>
        </div>



      <div class = "service-box">
        <img src="https://placehold.co/400x400" alt="Service 3 Badge" class="badge-icon">
        <p>Service  3 Description</p>
      </div>
</div> 


    </body>


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
      </style>
      <head>
        <title>
          Comments
        </title>
      </head>
      <body>
        <form action = "" method = "POST">
          <label>Name:
            <input type ="text" name = "Name" required /></label><br />
          <label>Comment:<br />
            <textarea name = "Comment" required ></textarea></label><br />
            <input type = "submit" name="Submit" value="Submit" />
          </form>

<?php
    if (isset($_POST["Submit"])) {
        print "<h2>Your comment has been submitted!</h2>";

        $Name = $_POST["Name"];
        $Comment = $_POST["Comment"];


        // Reading the old comments
        $Old = fopen("comments.txt", "r+t");



        $Old_Comments = fread($Old,1024);
     

        // Writing the new comment and appending the old ones
        $Write = fopen("comments.txt", "w+");
        if (!$Write) {
            die("Error: Unable to open the file for writing.");
        }

        $String = 
            "<div class='comment'><span>".$Name."</span><br />
            <span>".date("Y/m/d")." | ".date("h:i A")."</span><br />
            <span>".$Comment."</span></div>\n".$Old_Comments;

        fwrite($Write, $String);
        fclose($Write);
        fclose($Old);
      }


      if (file_exists("comments.txt")) {
        $Read = fopen("comments.txt", "r");
        if ($Read) {
            echo "<h1>Comments:</h1><hr>" . fread($Read, filesize("comments.txt"));
            fclose($Read);
        }
      }
?>
</body>
</html>