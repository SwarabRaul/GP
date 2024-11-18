using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerShooting : MonoBehaviour
{
    public Transform firePos;
    public GameObject bullet;
    public float timeBetweenShots;
    private bool canshoot = true;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if(Input.GetMouseButtonDown(0) && canshoot) {
            Shoot();
        }
        
    }

    public void Shoot() {
        Instantiate(bullet, firePos.position, firePos.rotation);
        StartCoroutine(ShootDelay());
    }

    IEnumerator ShootDelay() {
        canshoot = false;
        yield return new WaitForSeconds(timeBetweenShots);
        canshoot = true;
    }
}
