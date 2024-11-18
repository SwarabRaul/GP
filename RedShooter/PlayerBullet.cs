using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerBullet : MonoBehaviour
{
    public float bulletSpeed;
    private Rigidbody2D rb;

    private Player playerController;
    private GameObject playerobj;

    private void Awake() {
        Destroy(gameObject, 2f);
    }
    // Start is called before the first frame update
    void Start()
    {
        playerobj = GameObject.FindGameObjectWithTag("Player");
        playerController = playerobj.GetComponent<Player>();

        rb = GetComponent<Rigidbody2D>();
        // rb.velocity = transform.right * bulletSpeed;

        if(playerController.IsFacingRight()) {
            rb.velocity = transform.right * bulletSpeed;
        }
        else {
            rb.velocity = -transform.right * bulletSpeed;
        }
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnTriggerEnter2D(Collider2D collision) {
        if(collision.tag == "Enemy") {
            Destroy(gameObject);
            Destroy(collision.gameObject);
        }
    }

    private void OnCollisionEnter2D(Collision2D collision) {
        Destroy(gameObject);
    }
}
